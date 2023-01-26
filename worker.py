import argparse
import copy
import json
import os
import time

import torch
from oba import Obj
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from loader.data import Data
from loader.task.bart.bart4rec_task import Bart4RecTask
from loader.task.base_batch import BaseBatch
from loader.task.base_task import BaseTask
from loader.task.base_loss import TaskLoss, LossDepot
from loader.task.bert.bert4rec_task import Bert4RecTask
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask
from utils import metric
from utils.config_initializer import ConfigInitializer
from utils.dictifier import Dictifier
from utils.epoch import Epoch
from utils.gpu import GPU
from utils.monitor import Monitor
from utils.random_seed import seeding
from utils.smart_printer import SmartPrinter, printer, Color
from utils.logger import Logger
from utils.timer import Timer


class Worker:
    """
    Integrator, preparing, initializing, and accepting different tasks
    """

    def __init__(self, project_args, project_exp, cuda=None, display_batch=False):
        self.args = project_args
        self.exp = project_exp
        self.print = printer[('MAIN', '·', Color.CYAN)]

        self.display_batch = display_batch
        if self.display_batch:
            self.exp.policy.batch_size = 1

        self.logging = Logger(self.args.store.log_path)
        SmartPrinter.logger = self.logging

        self.device = self.get_device(cuda)

        self.data = Data(
            project_args=self.args,
            project_exp=self.exp,
            device=self.device,
        )

        self.print(self.data.depots['train'][0])

        self.auto_model = self.data.model(
            device=self.device,
            model_init=self.data.model_init,
            task_initializer=self.data.task_initializer,
        )

        self.auto_model.to(self.device)
        self.print(self.auto_model.model.config)
        self.save_model = self.auto_model
        self.disable_tqdm = bool(self.exp.display.disable_tqdm)

        self.static_modes = ['export', 'dev', 'test']
        self.in_static_modes = self.exp.mode in self.static_modes or self.exp.mode.startswith('test')

        if self.in_static_modes:
            self.m_optimizer = self.m_scheduler = None
        else:
            self.m_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.auto_model.parameters()),
                lr=self.exp.policy.lr
            )
            self.m_scheduler = get_linear_schedule_with_warmup(
                self.m_optimizer,
                num_warmup_steps=self.exp.policy.n_warmup,
                num_training_steps=len(self.data.train_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
            )

            self.print('training params')
            total_memory = 0
            cu_cluster_mlm = False
            for name, p in self.auto_model.named_parameters():  # type: str, torch.Tensor
                total_memory += p.element_size() * p.nelement()
                if 'cluster-mlm.decoder_layers' in name:
                    cu_cluster_mlm = True
                    continue
                if p.requires_grad and not name.startswith('bert.'):
                    self.print(name, p.data.shape, p.data.get_device())
            if cu_cluster_mlm:
                self.print('Ignore cluster-mlm decoder layers')
            self.print('total memory usage:', total_memory / 1024 / 8)

        if not self.exp.load.super_load:
            self.attempt_loading()

    @staticmethod
    def get_device(cuda):
        if cuda in [-1, False]:
            return 'cpu'
        if cuda is None:
            return GPU.auto_choose(torch_format=True)
        return "cuda:{}".format(cuda)

    def _attempt_loading(self, path):
        load_path = os.path.join(self.args.store.save_dir, path)
        while True:
            self.print("load model from exp {}".format(load_path))
            try:
                state_dict = torch.load(load_path, map_location=self.device)
                break
            except Exception as e:
                if not self.exp.load.wait_load:
                    raise e
                time.sleep(60)

        model_ckpt = state_dict['model']

        self.save_model.load_state_dict(model_ckpt, strict=not self.exp.load.relax_load)
        load_status = False
        if not self.in_static_modes and not self.exp.load.load_model_only:
            load_status = True
            self.m_optimizer.load_state_dict(state_dict['optimizer'])
            self.m_scheduler.load_state_dict(state_dict['scheduler'])
        self.print('Load optimizer and scheduler:', load_status)

    def attempt_loading(self):
        if self.exp.load.load_ckpt:
            self._attempt_loading(self.exp.load.load_ckpt)

    def log_interval(self, epoch, step, task: BaseTask, loss: TaskLoss):
        components = [f'step {step}', f'task {task.name}']
        for loss_name, loss_tensor in loss.get_loss_dict().items():
            if loss_name.endswith('loss') and isinstance(loss_tensor, torch.Tensor):
                components.append(f'{loss_name} {loss_tensor.item():.4f}')
        self.print[f'epoch {epoch}'](', '.join(components))

    def log_epoch(self, epoch, task: BaseTask, loss_depot: LossDepot):
        components = [f'task {task.name}']
        for loss_name, loss_value in loss_depot.depot.items():
            components.append(f'{loss_name} {loss_value:.4f}')
        self.print[f'epoch {epoch}'](', '.join(components))

    def train(self, *tasks: BaseTask):
        self.print('Start Training')
        if not self.exp.store:
            monitor = Monitor(
                ckpt_path=self.args.store.ckpt_path,
                interval=self.exp.policy.store_interval,
                super_store=False,
            )
        else:
            monitor = Monitor(
                ckpt_path=self.args.store.ckpt_path,
                **Obj.raw(self.exp.store)
            )

        train_steps = len(self.data.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        assert self.exp.policy.accumulate_batch >= 1

        # t_loader = self.data.get_t_loader(task)
        self.m_optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            loader = self.data.get_loader(self.data.TRAIN, *tasks).train()
            loader.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            self.auto_model.train()

            for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):  # type: int, BaseBatch
                task = batch.task
                task_output = self.auto_model(
                    batch=batch,
                    task=task,
                )

                loss = task.calculate_loss(batch, task_output, model=self.auto_model)
                loss.backward()

                accumulate_step += 1
                if accumulate_step == self.exp.policy.accumulate_batch:
                    self.m_optimizer.step()
                    self.m_scheduler.step()
                    self.m_optimizer.zero_grad()
                    accumulate_step = 0

                if self.exp.policy.check_interval:
                    if self.exp.policy.check_interval < 0:  # step part
                        if (step + 1) % max(train_steps // (-self.exp.policy.check_interval), 1) == 0:
                            self.log_interval(epoch, step, task, loss)
                    else:
                        if (step + 1) % self.exp.policy.check_interval == 0:
                            self.log_interval(epoch, step, task, loss)

            loss_depots = dict()
            for task in tasks:
                loss_depot = self.dev(task=task)
                self.log_epoch(epoch, task, loss_depot)
                loss_depots[task.name] = loss_depot.depot

            state_dict = dict(
                model=self.auto_model.state_dict(),
                optimizer=self.m_optimizer.state_dict(),
                scheduler=self.m_scheduler.state_dict(),
            )
            monitor.push(
                epoch=epoch,
                loss_depots=loss_depots,
                state_dict=state_dict,
            )

        self.print('Training Ended')
        monitor.export()

    def dev(self, task: BaseTask, steps=None):
        self.auto_model.eval()
        loader = self.data.get_loader(self.data.DEV, task).eval()
        loss_depot = LossDepot()

        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                task_output = self.auto_model(
                    batch=batch,
                    task=task,
                )

                loss = task.calculate_loss(batch, task_output, model=self.auto_model)
                loss_depot.add(loss)

            if steps and step >= steps:
                break

        return loss_depot.summarize()

    def test__curriculum_module_time(self, task: BaseTask, metric_pool):
        assert isinstance(task, BaseCurriculumMLMTask)
        steps = 0

        loader = self.data.get_loader(self.data.TEST, task).test()

        timer = Timer()
        self.auto_model.set_timer(timer)

        start_ = time.time()
        for batch in tqdm(loader):
            with torch.no_grad():
                output = self.auto_model(
                    batch=batch,
                    task=task,
                )
                start__ = time.time()
                task.t('curriculum', batch, output, metric_pool)
                end__ = time.time()
                timer.append('infer', end__ - start__)
            steps += 1
            if steps > 500:
                break
        end_ = time.time()
        self.print((end_ - start_) * 1000 / steps)
        timer.export()

        exit(0)

    def test__curriculum_time(self, task: BaseTask, metric_pool):
        assert isinstance(task, BaseCurriculumMLMTask)
        rounds = 5

        loader = self.data.get_loader(self.data.TEST, task).test()
        start = time.time()
        for _ in range(rounds):
            start_ = time.time()
            for batch in loader:
                with torch.no_grad():
                    output = self.auto_model(
                        batch=batch,
                        task=task,
                    )
                    task.t('curriculum', batch, output, metric_pool)
            end_ = time.time()
            self.print((end_ - start_) * 1000 / len(loader))

        end = time.time()
        self.print('Average Time:', (end - start) * 1000 / rounds / len(loader))
        exit(0)

    def test__curriculum(self, task: BaseTask, metric_pool: metric.MetricPool):
        assert isinstance(task, BaseCurriculumMLMTask)

        loader = self.data.get_loader(self.data.TEST, task).test()

        for batch in tqdm(loader, disable=self.disable_tqdm):
            with torch.no_grad():
                output = self.auto_model(
                    batch=batch,
                    task=task,
                )
                task.t('curriculum', batch, output, metric_pool)

    def test__left2right(self, task: BaseTask, metric_pool: metric.MetricPool):
        test_set = self.data.sets[self.data.TEST]
        test_depot = test_set.depot
        dictifier = Dictifier(aggregator=torch.stack)

        with torch.no_grad():
            index = 0
            samples = []
            for index in tqdm(range(*test_set.split_range), disable=self.disable_tqdm):
                sample = test_depot[index]
                sample = copy.deepcopy(sample)
                samples.append(sample)
                index += 1
                if index >= self.exp.policy.batch_size:
                    task.t('left2right', samples, self.auto_model, metric_pool, dictifier=dictifier)
                    index = 0
                    samples = []

            if index:
                task.t('left2right', samples, self.auto_model, metric_pool, dictifier=dictifier)

    def test__recall(self, task: BaseTask, metric_pool: metric.MetricPool):
        assert isinstance(task, Bert4RecTask)

        test_set = self.data.sets[self.data.TEST]
        test_depot = test_set.depot
        dictifier = Dictifier(aggregator=torch.stack)

        with torch.no_grad():
            index = 0
            samples = []
            for index in tqdm(range(*test_set.split_range), disable=self.disable_tqdm):
                sample = test_depot[index]
                sample = copy.deepcopy(sample)
                samples.append(sample)
                index += 1
                if index >= self.exp.policy.batch_size:
                    task.t('recall', samples, self.auto_model, metric_pool, dictifier=dictifier)
                    index = 0
                    samples = []

            if index:
                task.t('recall', samples, self.auto_model, metric_pool, dictifier=dictifier)

    def test__left2right(self, task: BaseTask, metric_pool: metric.MetricPool):
        assert isinstance(task, Bert4RecTask) or isinstance(task, Bart4RecTask)

        test_set = self.data.sets[self.data.TEST]
        test_depot = test_set.depot
        dictifier = Dictifier(aggregator=torch.stack)

        with torch.no_grad():
            index = 0
            samples = []
            for index in tqdm(range(*test_set.split_range), disable=self.disable_tqdm):
                sample = test_depot[index]
                sample = copy.deepcopy(sample)
                samples.append(sample)
                index += 1
                if index >= self.exp.policy.batch_size:
                    task.t('left2right', samples, self.auto_model, metric_pool, dictifier=dictifier)
                    index = 0
                    samples = []

            if index:
                task.t('left2right', samples, self.auto_model, metric_pool, dictifier=dictifier)

    def test_center(self, handler, task: BaseTask):
        metric_pool = metric.MetricPool()
        # metric_pool.add(metric.OverlapRate())
        metric_pool.add(metric.NDCG(), ns=self.exp.policy.n_metrics)
        metric_pool.add(metric.HitRate(), ns=self.exp.policy.n_metrics)
        metric_pool.init()

        self.auto_model.eval()
        task.test()

        handler(task, metric_pool)

        metric_pool.export()
        for metric_name, n in metric_pool.values:
            if n:
                self.print(f'{metric_name}@{n:4d}: {metric_pool.values[(metric_name, n)]:.4f}')
            else:
                self.print(f'{metric_name}     : {metric_pool.values[(metric_name, n)]:.4f}')

    def batch_displaying(self):
        tasks = self.data.tasks
        loader = self.data.get_loader(self.data.TRAIN, *tasks).train()
        loader.start_epoch(0, self.exp.policy.epoch)
        self.auto_model.train()

        for batch in loader:
            self.print(batch.export())
            return

    def run(self):
        if self.display_batch:
            self.batch_displaying()
            return

        # tasks = [self.data.pretrain_depot[task.name] for task in self.exp.tasks]
        tasks = self.data.tasks

        if self.exp.mode == 'train':
            self.train(*tasks)
        elif self.exp.mode == 'dev':
            dev_results = dict()
            for task in tasks:
                dev_results.update(self.dev(task, steps=100))
            display_string = []
            display_value = []
            for k in dev_results:
                display_string.append('%s {:.4f}' % k)
                display_value.append(dev_results[k])
            display_string = ', '.join(display_string)
            display_value = tuple(display_value)
            display_string = display_string.format(*display_value)
            self.print(display_string)
        elif self.exp.mode.startswith('test'):
            handler = object.__getattribute__(self, self.exp.mode)

            if not self.exp.load.super_load:
                for task in tasks:
                    if task.name == 'non':
                        continue
                    self.test_center(handler, task)
            else:
                interval, until, start = None, None, None
                if self.exp.load.auto_load:
                    epochs = json.load(open(os.path.join(
                        self.args.store.save_dir,
                        self.exp.load.ckpt_base_path,
                        'candidates.json'
                    )))
                else:
                    epochs = self.exp.load.epochs
                if isinstance(epochs, str):
                    epochs = eval(epochs)
                if not isinstance(epochs, list):
                    epochs, interval, until, start = None, epochs.interval, epochs.until, epochs.start
                epochs = Epoch(epochs, interval, until, start)

                while True:
                    epoch = epochs.next()
                    if epoch == -1:
                        break

                    ckpt_base_path = self.exp.load.ckpt_base_path
                    self._attempt_loading(os.path.join(ckpt_base_path, f'epoch_{epoch}.bin'))

                    for task in tasks:
                        if task.name == 'non':
                            continue
                        self.test_center(handler, task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--cuda', type=int, default=None)
    parser.add_argument('--display_batch', type=int, default=0)

    args = parser.parse_args()

    config, exp = ConfigInitializer.init(args.config, args.exp)
    seeding(2021)

    worker = Worker(
        project_args=config,
        project_exp=exp,
        cuda=args.cuda,
        display_batch=args.display_batch
    )
    worker.run()

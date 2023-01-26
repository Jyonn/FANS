import random
from typing import List

import torch

from loader.dataset.model_dataset import ModelDataset
from loader.init.caser_init import CaserInit
from loader.task.base_loss import TaskLoss
from loader.task.bert.bert4rec_task import Bert4RecBatch
from loader.task.bert.sasrec_task import SASRecTask, SASRecBatch


class CaserTask(SASRecTask):
    model_init: CaserInit
    name = 'caser'

    injecting_datasets = []  # type: List[ModelDataset]

    def __init__(self, **kwargs):
        super(CaserTask, self).__init__(**kwargs)

    def _injector_init(self, dataset: ModelDataset):
        super(CaserTask, self)._injector_init(dataset)
        CaserTask.injecting_datasets.append(dataset)

    def init(self, **kwargs):
        super(CaserTask, self).init(**kwargs)

        for dataset in CaserTask.injecting_datasets:
            dataset.max_sequence = self.model_init.max_length
            self.print('Inject max sequence:', dataset.mode)

    def produce_output(self, last_hidden_states, batch: Bert4RecBatch):
        return self._produce_output(last_hidden_states, batch)

    def sample_injector(self, sample):
        super(CaserTask, self).sample_injector(sample)

        if len(sample[self.concat_col]) >= self.model_init.max_length:
            start_index = random.randint(0, len(sample[self.concat_col]) - self.model_init.max_length)
            end_index = start_index + self.model_init.max_length
        else:
            start_index, end_index = 0, len(sample[self.concat_col])
        sample[self.concat_col] = sample[self.concat_col][start_index : end_index]

        if self.use_cluster:
            sample[self.cluster_col] = sample[self.cluster_col][start_index : end_index]
        return sample

    def calculate_loss(self, batch: SASRecBatch, output, **kwargs) -> TaskLoss:
        vocab_name = self.depot.get_vocab(self.concat_col)
        mask_labels = batch.append_info[self.select_col].to(self.device)
        loss = self.loss_fct(
            output[vocab_name],
            mask_labels
        )
        return TaskLoss(loss=loss)

    def test__left2right(self, samples, model, metric_pool, dictifier):
        ground_truths = []
        lengths = []

        arg_sorts = []
        for sample in samples:
            ground_truth = sample[self.p_global]
            lengths.append(len(sample[self.p_global]))
            sample[self.concat_col] = sample[self.k_global][-self.model_init.max_length:]
            sample[self.select_col] = -1
            if self.use_cluster:
                sample[self.cluster_col] = sample[self.k_cluster][-self.model_init.max_length:]
            ground_truths.append(ground_truth)
            arg_sorts.append([])

        for index in range(max(lengths)):
            batch = dictifier([self.dataset.build_format_data(sample) for sample in samples])
            batch = self._rebuild_batch(SASRecBatch(batch))

            outputs = model(
                batch=batch,
                task=self,
            )[self.depot.get_vocab(self.concat_col)]  # [B, S, V]

            for i_batch in range(len(samples)):
                arg_sort = torch.argsort(outputs[i_batch], descending=True).cpu().tolist()[
                           :metric_pool.max_n]
                arg_sorts[i_batch].append(arg_sort)
                samples[i_batch][self.concat_col] = samples[i_batch][self.concat_col][1:] + [arg_sort[0]]

                if self.use_cluster:
                    samples[i_batch][self.cluster_col] = samples[i_batch][self.cluster_col][1:] + [self.cluster_map[arg_sort[0]]]

        for i_batch, sample in enumerate(samples):
            candidates = []
            for depth in range(metric_pool.max_n):
                for index in range(len(sample[self.p_global])):
                    if arg_sorts[i_batch][index][depth] not in candidates:
                        candidates.append(arg_sorts[i_batch][index][depth])
                if len(candidates) >= metric_pool.max_n:
                    break

            metric_pool.push(candidates, ground_truths[i_batch])

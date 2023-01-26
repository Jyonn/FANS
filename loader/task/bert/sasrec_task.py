import random

import torch

from loader.dataset.order import Order
from loader.task.base_loss import TaskLoss
from loader.task.bert.bert4rec_task import Bert4RecTask, Bert4RecBatch
from loader.task.utils.base_mlm_task import MLMBertBatch
from utils.transformers_adaptor import BertOutput


class SASRecBatch(MLMBertBatch):
    def __init__(self, batch):
        super(SASRecBatch, self).__init__(batch=batch)
        self.mask_index = None

        self.register('mask_index')


class SASRecTask(Bert4RecTask):
    """
    MLM task for ListCont
    """

    name = 'sasrec'
    batcher = SASRecBatch

    def __init__(self, **kwargs):
        super(SASRecTask, self).__init__(**kwargs)

        self.select_col = 'select'

    def _injector_init(self, dataset):
        super(SASRecTask, self)._injector_init(dataset=dataset)
        dataset.append_cols.append(self.select_col)

    def sample_injector(self, sample):
        sample = super().sample_injector(sample=sample)
        if self.is_testing:
            select_index = len(sample[self.select_col]) - 1
        else:
            select_index = random.choice(range(len(sample[self.concat_col]) - 1))
        sample[self.select_col] = sample[self.concat_col][select_index]
        sample[self.concat_col] = sample[self.concat_col][:select_index]
        if self.use_cluster:
            sample[self.cluster_col] = sample[self.cluster_col][:select_index]
        return sample

    def _rebuild_batch(self, batch: SASRecBatch):
        mask_index = []
        for i_batch in range(batch.batch_size):
            col_end = self.dataset.max_sequence - 1
            for i_tok in range(self.dataset.max_sequence - 1, -1, -1):
                if batch.col_mask[self.concat_col][i_batch][i_tok]:
                    col_end = i_tok
                    break
            mask_index.append(col_end)

        batch.mask_index = torch.tensor(mask_index)

        return batch

    def produce_output(self, model_output: BertOutput, batch: Bert4RecBatch):
        return self._produce_output(model_output.last_hidden_state, batch)

    def calculate_loss(self, batch: SASRecBatch, output, **kwargs) -> TaskLoss:
        vocab_name = self.depot.get_vocab(self.concat_col)
        vocab_size = self.depot.get_vocab_size(self.concat_col)
        sequence_len = int(output[vocab_name].shape[1])

        mask_index = batch.mask_index.unsqueeze(dim=-1).long().to(self.device)
        masked_elements = torch.zeros(batch.batch_size, sequence_len, dtype=torch.bool).to(self.device)
        trues = torch.ones(batch.batch_size, sequence_len, dtype=torch.bool).to(self.device)
        masked_elements.scatter_(1, mask_index, trues)

        distribution = torch.masked_select(
            output[vocab_name], masked_elements.unsqueeze(dim=-1)).view(-1, vocab_size).to(self.device)

        mask_labels = batch.append_info[self.select_col].to(self.device)
        loss = self.loss_fct(
            distribution,
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
            sample[self.concat_col] = sample[self.k_global][:]
            sample[self.select_col] = -1
            if self.use_cluster:
                sample[self.cluster_col] = sample[self.k_cluster][:]
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
                mask_index = batch.mask_index[i_batch]

                arg_sort = torch.argsort(outputs[i_batch][mask_index], descending=True).cpu().tolist()[
                           :metric_pool.max_n]
                arg_sorts[i_batch].append(arg_sort)
                samples[i_batch][self.concat_col].append(arg_sort[0])
                if self.use_cluster:
                    samples[i_batch][self.cluster_col].append(self.cluster_map[arg_sort[0]])

        for i_batch, sample in enumerate(samples):
            candidates = []
            for depth in range(metric_pool.max_n):
                for index in range(len(sample[self.p_global])):
                    if arg_sorts[i_batch][index][depth] not in candidates:
                        candidates.append(arg_sorts[i_batch][index][depth])
                if len(candidates) >= metric_pool.max_n:
                    break

            metric_pool.push(candidates, ground_truths[i_batch])

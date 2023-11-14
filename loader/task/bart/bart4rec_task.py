import copy
import json
import os

import numpy as np
import torch
from UniTok import Vocab

from loader.dataset.bart_dataset import BartDataset
from loader.dataset.order import Order
from loader.task.utils.base_classifiers import BartClassifier
from loader.task.utils.base_mlm_task import MLMBartBatch, BaseMLMTask
from utils.transformers_adaptor import BartOutput


class Bart4RecBatch(MLMBartBatch):
    def __init__(self, batch):
        super(Bart4RecBatch, self).__init__(batch=batch)
        self.mask_index = None

        self.register('mask_index')


class Bart4RecTask(BaseMLMTask):
    """
    MLM task for ListCont
    """

    name = 'bart4rec'
    mask_scheme = 'MASK'
    dataset: BartDataset
    cls_module = BartClassifier
    batcher = Bart4RecBatch
    injection = ['train', 'dev']

    def __init__(
            self,
            k_global='k_global',
            p_global='p_global',
            k_cluster='k_cluster',
            p_cluster='p_cluster',
            mask_last_ratio: float = 0.1,
            use_cluster=False,
            cluster_json=None,
            **kwargs,
    ):
        super(Bart4RecTask, self).__init__(**kwargs)

        self.k_global = k_global
        self.p_global = p_global
        self.k_cluster = k_cluster
        self.p_cluster = p_cluster
        self.use_cluster = use_cluster

        self.concat_col = 'list'
        self.cluster_col = 'cluster'
        self.col_order = [self.concat_col]
        self.cluster_json = cluster_json

        self.mask_last_ratio = mask_last_ratio

    # rebuild sample in dataset layer by init and dataset_injector

    def init(self, **kwargs):
        super().init(**kwargs)
        self.depot.col_info[self.concat_col] = dict(vocab=self.depot.col_info[self.k_global].vocab)

        if self.use_cluster:
            self.depot.col_info[self.cluster_col] = dict(vocab=self.depot.col_info[self.k_cluster].vocab)

            self.cluster_vocab_count = json.load(open(os.path.join(self.depot.store_dir, self.cluster_json)))
            self.n_clusters = len(self.cluster_vocab_count)

            cluster_vocabs = self._load_cluster_vocabs()
            global_vocab = self.depot.vocab_depot(self.depot.get_vocab(self.p_global))
            self.cluster_map = [0] * global_vocab.get_size()  # type: list
            for i_cluster, vocab in enumerate(cluster_vocabs):  # type: Vocab
                for index in range(vocab.get_size()):
                    index = global_vocab.obj2index[vocab.index2obj[index]]  # type: int
                    self.cluster_map[index] = i_cluster

    def _load_cluster_vocabs(self):
        vocab_path = os.path.dirname(os.path.join(self.depot.store_dir, self.cluster_json))
        return [Vocab(f'cluster_{i}').load(vocab_path) for i in range(self.n_clusters)]

    def _injector_init(self, dataset):
        # not only one dataset is required to be initialized
        if self.use_cluster:
            dataset.order = Order([[self.concat_col, self.cluster_col]])
        else:
            dataset.order = Order([self.concat_col])

    def sample_injector(self, sample):
        sample[self.concat_col] = sample[self.k_global] + sample[self.p_global]
        del sample[self.k_global], sample[self.p_global]

        if self.use_cluster:
            sample[self.cluster_col] = sample[self.k_cluster] + sample[self.p_cluster]
            del sample[self.k_cluster], sample[self.p_cluster]
        return sample

    def prepare_batch(self, batch: MLMBartBatch):
        batch.encoder.mask_labels = torch.ones(batch.batch_size, self.dataset.max_sequence, dtype=torch.long) * self.loss_pad
        batch.encoder.mask_labels_col = copy.deepcopy(batch.encoder.col_mask)

    def _rebuild_batch(self, batch: Bart4RecBatch):
        self.prepare_batch(batch)

        mask_last = np.random.uniform() < self.mask_last_ratio

        if mask_last or not self.is_training:
            mask_index = []

            for i_batch in range(batch.batch_size):
                col_end = None
                for i_tok in range(self.dataset.max_sequence - 1, -1, -1):
                    if batch.encoder.col_mask[self.concat_col][i_batch][i_tok]:
                        col_end = i_tok
                        break
                mask_index.append(col_end)

                batch.encoder.mask_labels[i_batch][col_end] = batch.encoder.input_ids[i_batch][col_end]
                batch.encoder.input_ids[i_batch][col_end] = self.dataset.TOKENS[self.mask_scheme]
                batch.encoder.col_mask[self.concat_col][i_batch][col_end] = 0
                batch.encoder.col_mask[self.dataset.special_id][i_batch][col_end] = 1
            if self.is_testing:
                batch.mask_index = torch.tensor(mask_index)
        else:
            self.random_mask(batch.encoder, self.concat_col)

        return batch

    def produce_output(self, model_output: BartOutput, batch: Bart4RecBatch):
        return self._produce_output(model_output.decoder_hidden_states, batch)

    def test__left2right(self, samples, model, metric_pool, dictifier):
        ground_truths = []
        lengths = []

        arg_sorts = []
        for sample in samples:
            ground_truth = sample[self.p_global]
            lengths.append(len(sample[self.p_global]))
            sample[self.concat_col] = sample[self.k_global][:]
            if self.use_cluster:
                sample[self.cluster_col] = sample[self.k_cluster][:]
            ground_truths.append(ground_truth)
            arg_sorts.append([])

        for index in range(max(lengths)):
            for sample in samples:
                sample[self.concat_col].append(0)
                if self.use_cluster:
                    sample[self.cluster_col].append(0)
            batch = dictifier([self.dataset.build_format_data(sample) for sample in samples])
            batch = self._rebuild_batch(Bart4RecBatch(batch))

            outputs = model(
                batch=batch,
                task=self,
            )[self.depot.get_vocab(self.concat_col)]  # [B, S, V]

            for i_batch in range(len(samples)):
                mask_index = batch.mask_index[i_batch]

                arg_sort = torch.argsort(outputs[i_batch][mask_index], descending=True).cpu().tolist()[:metric_pool.max_n]
                arg_sorts[i_batch].append(arg_sort)
                samples[i_batch][self.concat_col][-1] = arg_sort[0]
                if self.use_cluster:
                    samples[i_batch][self.cluster_col][-1] = self.cluster_map[arg_sort[0]]

        for i_batch, sample in enumerate(samples):
            candidates = []
            for depth in range(metric_pool.max_n):
                for index in range(len(sample[self.p_global])):
                    if arg_sorts[i_batch][index][depth] not in candidates:
                        candidates.append(arg_sorts[i_batch][index][depth])
                if len(candidates) >= metric_pool.max_n:
                    break

            metric_pool.push(candidates, ground_truths[i_batch])

    def test__recall(self, samples, model, metric_pool, dictifier):
        ground_truths = []
        lengths = []

        arg_sorts = []
        for sample in samples:
            ground_truth = sample[self.p_global]
            lengths.append(len(sample[self.p_global]))
            sample[self.concat_col] = sample[self.k_global][:]
            sample[self.concat_col].append(0)
            if self.use_cluster:
                sample[self.cluster_col] = sample[self.k_cluster][:]
                sample[self.cluster_col].append(0)
            ground_truths.append(ground_truth)
            arg_sorts.append([])

        batch = dictifier([self.dataset.build_format_data(sample) for sample in samples])
        batch = self._rebuild_batch(Bart4RecBatch(batch))

        outputs = model(
            batch=batch,
            task=self,
        )[self.depot.get_vocab(self.concat_col)]  # [B, S, V]

        for i_batch in range(len(samples)):
            mask_index = batch.mask_index[i_batch]

            candidates = torch.argsort(outputs[i_batch][mask_index], descending=True).cpu().tolist()[:lengths[i_batch]]
            metric_pool.push(candidates, ground_truths[i_batch])

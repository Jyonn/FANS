import json
import os.path
from abc import ABC
from typing import Union

import torch
from UniTok import Vocab
from torch import nn

from loader.task.base_loss import TaskLoss
from loader.task.utils.base_classifiers import BertClusterClassifier, BartClusterClassifier, BertClassifier, \
    BartClassifier
from loader.task.utils.base_mlm_task import BaseMLMTask, MLMBertBatch


class ClusterMLMTaskLoss(TaskLoss):
    def __init__(self, local_loss, cluster_loss):
        super(ClusterMLMTaskLoss, self).__init__(loss=local_loss + cluster_loss)
        self.local_loss = local_loss
        self.cluster_loss = cluster_loss

    # def backward(self):
    #     loss = self.loss + self.cluster_loss
    #     if loss.requires_grad:
    #         loss.backward()


class BaseClusterMLMTask(BaseMLMTask, ABC):
    name = 'base-cluster-mlm'
    cluster_cls_module: Union[BertClusterClassifier, BartClusterClassifier]
    cls_module: Union[BertClassifier, BartClassifier]

    def __init__(
            self,
            cluster_json,
            k_global='k_global',
            p_global='p_global',
            k_local='k_local',
            p_local='p_local',
            k_cluster='k_cluster',
            p_cluster='p_cluster',
            grad_cluster_loss=True,
            grad_local_loss=True,
            **kwargs
    ):
        super(BaseClusterMLMTask, self).__init__(**kwargs)

        self.k_global = k_global
        self.k_local = k_local
        self.k_cluster = k_cluster
        self.p_global = p_global
        self.p_local = p_local
        self.p_cluster = p_cluster
        self.grad_cluster_loss = grad_cluster_loss
        self.grad_local_loss = grad_local_loss

        self.cluster_json = cluster_json
        self.col_cluster_dict = {
            self.k_global: self.k_cluster,
            self.p_global: self.p_cluster
        }

        self.col_pairs = [(self.k_cluster, self.k_local), (self.p_cluster, self.p_local)]

    def _load_cluster_vocabs(self):
        vocab_path = os.path.dirname(os.path.join(self.depot.store_dir, self.cluster_json))
        return [Vocab(f'cluster_{i}').load(vocab_path) for i in range(self.n_clusters)]

    def init(self, **kwargs):
        super().init(**kwargs)

        self.cluster_vocab_count = json.load(open(os.path.join(self.depot.store_dir, self.cluster_json)))
        self.n_clusters = len(self.cluster_vocab_count)

        cluster_vocabs = self._load_cluster_vocabs()
        global_vocab = self.depot.vocab_depot(self.depot.get_vocab(self.p_global))
        self.local_global_maps = []
        for vocab in cluster_vocabs:  # type: Vocab
            map_ = []
            for index in range(vocab.get_size()):
                map_.append(global_vocab.obj2index[vocab.index2obj[index]])
            self.local_global_maps.append(map_)

    def get_embedding(self, **kwargs):
        return super().get_embedding(**kwargs, enable_attrs={self.k_global, self.p_global})

    def _init_extra_module(self):
        return nn.ModuleDict(dict(
            cluster_cls=self.cluster_cls_module.create(
                key='cluster',
                cluster_vocabs=self.cluster_vocab_count,
                config=self.model_init.model_config,
            ),
            cls=self.cls_module.create(
                config=self.model_init.model_config,
                key=self.depot.get_vocab(self.k_cluster),
                vocab_size=self.depot.get_vocab_size(self.k_cluster)
            )
        ))

    def _produce_output(self, last_hidden_state, batch: MLMBertBatch):
        mask_labels = batch.mask_labels.to(self.device)  # type: torch.Tensor

        output_dict = dict()

        cls_module = self.extra_module['cls']
        cluster_cls_module = self.extra_module['cluster_cls']

        output_dict['pred_cluster_distribution'] = pred_clusters = cls_module(last_hidden_state)  # [B, N, V]
        output_dict['pred_cluster_labels'] = pred_cluster_labels = torch.argmax(pred_clusters.detach(), dim=-1).to(self.device)

        for col_name, local_col_name in self.col_pairs:
            if col_name not in batch.mask_labels_col:
                continue
            mask_labels_col = batch.mask_labels_col[col_name].to(self.device)
            col_mask = batch.col_mask[col_name].to(self.device)
            masked_elements = torch.not_equal(col_mask, mask_labels_col)

            if not self.is_testing:
                current_cluster_labels = masked_elements * (mask_labels + 1)
                current_pred_cluster_labels = masked_elements * (pred_cluster_labels + 1)
                cluster_labels = torch.eq(current_cluster_labels, current_pred_cluster_labels) * current_cluster_labels - 1
            else:
                cluster_labels = masked_elements * (pred_cluster_labels + 1) - 1

            output_dict[local_col_name] = cluster_cls_module(
                last_hidden_state,
                cluster_labels=cluster_labels,
            )
        return output_dict

    def calculate_loss(self, batch: MLMBertBatch, output, **kwargs) -> ClusterMLMTaskLoss:
        weight = kwargs.get('weight', 1)
        mask_labels = batch.mask_labels.to(self.device)  # type: torch.Tensor

        total_cluster_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        total_local_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        for col_name, local_col_name in self.col_pairs:
            if col_name not in batch.mask_labels_col:
                continue

            vocab_size = self.depot.get_vocab_size(col_name)
            mask_labels_col = batch.mask_labels_col[col_name].to(self.device)  # type: torch.Tensor

            col_mask = batch.col_mask[col_name].to(self.device)
            masked_elements = torch.not_equal(col_mask, mask_labels_col)  # type: torch.Tensor
            if not torch.sum(masked_elements):
                continue

            if self.grad_cluster_loss:
                distribution = torch.masked_select(
                    output['pred_cluster_distribution'], masked_elements.unsqueeze(dim=-1)
                ).view(-1, vocab_size).to(self.device)
                col_labels = torch.masked_select(mask_labels, masked_elements).to(self.device)

                loss = self.loss_fct(
                    distribution,
                    col_labels
                )
                total_cluster_loss += loss * weight

            if self.grad_local_loss:
                cluster_labels = masked_elements * (mask_labels + 1)
                pred_cluster_labels = masked_elements * (output['pred_cluster_labels'] + 1)
                cluster_labels = torch.eq(cluster_labels, pred_cluster_labels) * cluster_labels - 1

                local_labels = batch.attr_ids[col_name][local_col_name].to(self.device)
                for i_cluster in range(self.n_clusters):
                    if not (cluster_labels == i_cluster).sum():
                        continue

                    cluster_masked_elements = ((cluster_labels == i_cluster) * masked_elements).to(self.device)
                    i_cluster_labels = torch.masked_select(local_labels, cluster_masked_elements).to(self.device)  # [B, K]
                    loss = self.loss_fct(
                        output[local_col_name][i_cluster],  # [B, K, V]
                        i_cluster_labels,  # [B, K]
                    )
                    total_local_loss += loss * weight / self.n_clusters
        # exit(0)
        return ClusterMLMTaskLoss(local_loss=total_local_loss, cluster_loss=total_cluster_loss)

    def test__curriculum(self, batch: MLMBertBatch, output, metric_pool):
        mask_labels_col = batch.mask_labels_col
        indexes = batch.append_info['index']

        pred_cluster_labels = output['pred_cluster_labels']
        output = output[self.p_local]
        col_mask = mask_labels_col[self.p_cluster]

        cluster_indexes = [0] * self.n_clusters

        for i_batch in range(len(indexes)):
            arg_sorts = []
            for i_tok in range(self.dataset.max_sequence):
                if col_mask[i_batch][i_tok]:
                    cluster_id = pred_cluster_labels[i_batch][i_tok]
                    top_items = torch.argsort(
                        output[cluster_id][cluster_indexes[cluster_id]], descending=True
                    ).cpu().tolist()[:metric_pool.max_n]
                    top_items = [self.local_global_maps[cluster_id][item] for item in top_items]
                    arg_sorts.append(top_items)
                    cluster_indexes[cluster_id] += 1
                else:
                    arg_sorts.append(None)

            ground_truth = self.depot.pack_sample(indexes[i_batch])[self.p_global]
            candidates = []
            for depth in range(metric_pool.max_n):
                for i_tok in range(self.dataset.max_sequence):
                    if col_mask[i_batch][i_tok]:
                        if arg_sorts[i_tok][depth] not in candidates:
                            candidates.append(arg_sorts[i_tok][depth])
                if len(candidates) >= metric_pool.max_n:
                    break

            metric_pool.push(candidates, ground_truth)

        for cluster_id in range(self.n_clusters):
            if output[cluster_id] is None:
                assert not cluster_indexes[cluster_id]
            else:
                assert cluster_indexes[cluster_id] == len(output[cluster_id])

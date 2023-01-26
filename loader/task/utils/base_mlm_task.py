import copy
from abc import ABC
from typing import Union, Optional, Type, Dict

import numpy as np
import torch
from torch import nn

from loader.dataset.order import Order
from loader.task.base_batch import BertBatch, BartBatch
from loader.task.base_task import BaseTask
from loader.task.base_loss import TaskLoss
from loader.task.utils.base_classifiers import BartClassifier, BertClassifier


class MLMBertBatch(BertBatch):
    def __init__(self, batch):
        super(MLMBertBatch, self).__init__(batch=batch)
        self.mask_labels = None  # type: Optional[torch.Tensor]
        self.mask_labels_col = None  # type: Optional[Dict[str, torch.Tensor]]
        self.mask_ratio = None  # type: Optional[float]

        self.register('mask_labels', 'mask_labels_col', 'mask_ratio')


class MLMBartBatch(BartBatch):
    batcher = MLMBertBatch

    def __init__(self, batch):
        super().__init__(batch)
        self.batch_size = self.encoder.batch_size


class BaseMLMTask(BaseTask, ABC):
    name = 'base-mlm'
    mask_scheme = 'MASK_{col}'
    mask_col_ph = '{col}'
    cls_module: Union[Type[BertClassifier], Type[BartClassifier]]
    col_order: list
    batcher: Union[Type[MLMBertBatch], Type[MLMBartBatch]]

    def __init__(
            self,
            select_prob=0.15,
            mask_prob=0.8,
            random_prob=0.1,
            loss_pad=-100,
            apply_cols=None,
            **kwargs
    ):
        super(BaseMLMTask, self).__init__()

        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.loss_pad = loss_pad

        self.apply_cols = apply_cols
        self.loss_fct = nn.CrossEntropyLoss()

    def get_col_order(self, order: Order):
        order = list(map(lambda x: x.name, order.order))
        if not self.apply_cols:
            return copy.deepcopy(order)
        return list(filter(lambda col: col in self.apply_cols, order))

    def get_expand_tokens(self):
        return [self.mask_scheme]

    def get_mask_token(self, col_name):
        return self.dataset.TOKENS[self.mask_scheme.replace(self.mask_col_ph, col_name)]

    def prepare_batch(self, batch: MLMBertBatch):
        batch.mask_labels = torch.ones(batch.batch_size, self.dataset.max_sequence, dtype=torch.long) * self.loss_pad
        batch.mask_labels_col = copy.deepcopy(batch.col_mask)

    def do_mask(self, mask, tok, vocab_size):
        tok = int(tok)

        if np.random.uniform() < self.select_prob:
            mask_type = np.random.uniform()
            if mask_type < self.mask_prob:
                return mask, tok, True
            elif mask_type < self.mask_prob + self.random_prob:
                return np.random.randint(vocab_size), tok, False
            return tok, tok, False
        return tok, self.loss_pad, False

    def random_mask(self, batch: MLMBertBatch, col_name):
        vocab_size = self.depot.get_vocab_size(col_name)

        for i_batch in range(batch.batch_size):
            for i_tok in range(self.dataset.max_sequence):
                if batch.col_mask[col_name][i_batch][i_tok]:
                    input_id, mask_label, use_special_col = self.do_mask(
                        mask=self.get_mask_token(col_name),
                        tok=batch.input_ids[i_batch][i_tok],
                        vocab_size=vocab_size
                    )
                    batch.input_ids[i_batch][i_tok] = input_id
                    batch.mask_labels[i_batch][i_tok] = mask_label
                    if use_special_col:
                        batch.col_mask[col_name][i_batch][i_tok] = 0
                        batch.col_mask[self.dataset.special_id][i_batch][i_tok] = 1

    def left2right_mask(self, batch: MLMBertBatch, col_name):
        for i_batch in range(batch.batch_size):
            col_start, col_end = None, None
            for i_tok in range(self.dataset.max_sequence):
                if batch.col_mask[col_name][i_batch][i_tok]:
                    if col_start is None:
                        col_start = i_tok
                    else:
                        col_end = i_tok
            col_end += 1

            if self.is_training:
                mask_count = int((col_end - col_start) * batch.mask_ratio)
                col_start = col_end - mask_count

            selected_tokens = slice(col_start, col_end)

            batch.mask_labels[i_batch][selected_tokens] = batch.input_ids[i_batch][selected_tokens]
            batch.input_ids[i_batch][selected_tokens] = self.get_mask_token(col_name)
            batch.col_mask[col_name][i_batch][selected_tokens] = 0
            batch.col_mask[self.dataset.special_id][i_batch][selected_tokens] = 1

    def _init_extra_module(self):
        module_dict = dict()

        for col_name in self.col_order:
            vocab = self.depot.col_info[col_name].vocab

            self.print(f'preparing CLS module for {col_name} - {vocab}')
            if vocab in module_dict:
                self.print(f'exist in module dict, skip')
                continue

            vocab_size = self.depot.get_vocab_size(vocab, as_vocab=True)
            module_dict[vocab] = self.cls_module.create(
                config=self.model_init.model_config,
                key=vocab,
                vocab_size=vocab_size,
            )
            self.print(f'created')
        return nn.ModuleDict(module_dict)

    def _produce_output(self, last_hidden_state, batch: Union[MLMBertBatch, MLMBartBatch]):
        output_dict = dict()
        for vocab_name in self.extra_module:
            classification_module = self.extra_module[vocab_name]
            output_dict[vocab_name] = classification_module(last_hidden_state)
        return output_dict

    def calculate_loss(self, batch: MLMBertBatch, output, **kwargs) -> TaskLoss:
        # weight = kwargs.get('weight', 1)
        #
        # mask_labels = batch.mask_labels.to(self.device)  # type: torch.Tensor
        #
        # total_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        # for col_name in self.col_order:
        #     vocab_name = self.depot.get_vocab(col_name)
        #     mask_labels_col = batch.mask_labels_col[col_name].to(self.device)  # type: torch.Tensor
        #     col_mask = batch.col_mask[col_name].to(self.device)
        #     masked_elements = torch.not_equal(col_mask, mask_labels_col)  # type: torch.Tensor
        #     if not torch.sum(masked_elements):
        #         continue
        #
        #     col_labels = torch.mul(mask_labels_col, mask_labels) + \
        #                  torch.ones(mask_labels.shape, dtype=torch.long).to(self.device) * (mask_labels_col - 1) * 100
        #     col_labels = col_labels.view(-1).to(self.device)
        #     vocab_size = self.depot.get_vocab_size(col_name)
        #     loss = self.loss_fct(
        #         output[vocab_name].view(-1, vocab_size),
        #         col_labels
        #     )
        #     total_loss += loss * weight
        # return TaskLoss(loss=total_loss)

        weight = kwargs.get('weight', 1)

        mask_labels = batch.mask_labels.to(self.device)  # type: torch.Tensor

        total_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        for col_name in self.col_order:
            vocab_name = self.depot.get_vocab(col_name)
            vocab_size = self.depot.get_vocab_size(col_name)

            mask_labels_col = batch.mask_labels_col[col_name].to(self.device)  # type: torch.Tensor

            col_mask = batch.col_mask[col_name].to(self.device)
            masked_elements = torch.not_equal(col_mask, mask_labels_col)  # type: torch.Tensor
            if not torch.sum(masked_elements):
                continue

            distribution = torch.masked_select(
                output[vocab_name], masked_elements.unsqueeze(dim=-1)).view(-1, vocab_size).to(self.device)
            col_labels = torch.masked_select(mask_labels, masked_elements).to(self.device)

            loss = self.loss_fct(
                distribution,
                col_labels
            )
            total_loss += loss * weight
        return TaskLoss(loss=total_loss)

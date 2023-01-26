import time
from typing import Union

import torch

from loader.dataset.bert_dataset import BertDataset
from loader.init.bert_init import BertInit
from loader.task.base_batch import BertBatch
from model.auto_model import AutoModel
from utils.transformers_adaptor import BertModel, BertOutput

from loader.task.base_task import BaseTask


class AutoBert(AutoModel):
    model: BertModel
    dataset_class = BertDataset
    model_initializer = BertInit

    def __init__(self, **kwargs):
        super(AutoBert, self).__init__(model_class=BertModel, **kwargs)

    def forward(self, batch: BertBatch, task: Union[str, BaseTask]):
        attention_mask = batch.attention_mask.to(self.device)  # type: torch.Tensor # [B, S]
        segment_ids = batch.segment_ids.to(self.device)  # type: torch.Tensor # [B, S]

        if isinstance(task, str):
            task = self.task_initializer[task]

        input_embeds = task.get_embedding(
            batch=batch,
            table_dict=self.embedding_tables,
            embedding_size=self.hidden_size,
        )

        start_ = time.time()
        bert_output = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            token_type_ids=segment_ids,
            output_hidden_states=True,
            return_dict=True
        )  # type: BertOutput
        end_ = time.time()
        if self.timer:
            self.timer.append('model', end_ - start_)

        # self.print('bert output device', bert_output.last_hidden_state.get_device())

        start_ = time.time()
        output = task.produce_output(bert_output, batch=batch)
        end_ = time.time()
        if self.timer:
            self.timer.append('output', end_ - start_)
        return output

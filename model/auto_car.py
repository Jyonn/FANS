from typing import Union

import torch

from loader.dataset.bert_dataset import BertDataset
from loader.init.bert_init import BertInit
from loader.task.base_batch import BertBatch
from model.auto_model import AutoModel
from model.network.car import CarModel

from loader.task.base_task import BaseTask


class AutoCar(AutoModel):
    model: CarModel
    dataset_class = BertDataset
    model_initializer = BertInit

    def __init__(self, **kwargs):
        super(AutoCar, self).__init__(model_class=CarModel, **kwargs)

    def forward(self, batch: BertBatch, task: Union[str, BaseTask]):
        attention_mask = batch.attention_mask.to(self.device)  # type: torch.Tensor # [B, S]

        if isinstance(task, str):
            task = self.task_initializer[task]

        input_embeds = task.get_embedding(
            batch=batch,
            table_dict=self.embedding_tables,
            embedding_size=self.hidden_size,
        )

        last_hidden_states = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )

        return task.produce_output(last_hidden_states, batch=batch)

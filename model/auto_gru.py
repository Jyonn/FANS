from typing import Union

from loader.dataset.bert_dataset import BertDataset
from loader.init.gru_init import GruInit
from loader.task.base_batch import BertBatch
from model.auto_model import AutoModel

from loader.task.base_task import BaseTask
from model.network.gru import GruModel


class AutoGru(AutoModel):
    model: GruModel
    dataset_class = BertDataset
    model_initializer = GruInit

    def __init__(self, **kwargs):
        super(AutoGru, self).__init__(model_class=GruModel, **kwargs)

    def forward(self, batch: BertBatch, task: Union[str, BaseTask]):
        if isinstance(task, str):
            task = self.task_initializer[task]

        input_embeds = task.get_embedding(
            batch=batch,
            table_dict=self.embedding_tables,
            embedding_size=self.hidden_size,
        )

        last_hidden_states = self.model(
            inputs_embeds=input_embeds,
        )

        return task.produce_output(last_hidden_states, batch=batch)

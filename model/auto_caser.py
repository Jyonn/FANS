from typing import Union

from loader.dataset.bert_dataset import BertDataset
from loader.init.caser_init import CaserInit
from loader.task.base_batch import BertBatch
from model.auto_model import AutoModel

from loader.task.base_task import BaseTask
from model.network.caser import CaserModel


class AutoCaser(AutoModel):
    model: CaserModel
    dataset_class = BertDataset
    model_initializer = CaserInit

    def __init__(self, **kwargs):
        super(AutoCaser, self).__init__(model_class=CaserModel, **kwargs)

    def forward(self, batch: BertBatch, task: Union[str, BaseTask]):
        if isinstance(task, str):
            task = self.task_initializer[task]

        input_embeds = task.get_embedding(
            batch=batch,
            table_dict=self.embedding_tables,
            embedding_size=self.hidden_size,
        )

        output = self.model(
            inputs_embeds=input_embeds,
        )

        return task.produce_output(output, batch=batch)

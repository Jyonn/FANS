import random
from typing import List

from torch.utils.data import DataLoader

from loader.dataset.model_dataset import ModelDataset
from loader.task.base_batch import BaseBatch
from loader.task.base_task import BaseTask


class ModelDataLoader(DataLoader):
    def __init__(self, dataset: ModelDataset, tasks: List[BaseTask], **kwargs):
        super().__init__(
            dataset=dataset,
            **kwargs
        )

        self.auto_dataset = dataset
        self.tasks = tasks

    def start_epoch(self, current_epoch, total_epoch):
        for task in self.tasks:
            task.start_epoch(current_epoch, total_epoch)
        return self

    def test(self):
        for task in self.tasks:
            task.test()
        return self

    def eval(self):
        for task in self.tasks:
            task.eval()
        return self

    def train(self):
        for task in self.tasks:
            task.train()
        return self

    def __iter__(self):
        iterator = super().__iter__()

        while True:
            try:
                batch = next(iterator)
                task = random.choice(self.tasks)
                batch = task.rebuild_batch(batch)  # type: BaseBatch
                batch.task = task
                yield batch
            except StopIteration:
                return

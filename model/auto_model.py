from typing import Union, Type, Optional

from torch import nn
from transformers import PreTrainedModel

from loader.dataset.model_dataset import ModelDataset
from loader.init.model_init import ModelInit

from loader.task.task_initializer import TaskInitializer
from loader.task.base_task import BaseTask

from utils.smart_printer import printer, Color
from utils.timer import Timer


class AutoModel(nn.Module):
    model: PreTrainedModel
    dataset_class: Type[ModelDataset]
    model_initializer: Type[ModelInit]

    def __init__(
            self,
            model_init: ModelInit,
            device,
            task_initializer: TaskInitializer,
            model_class: callable,
    ):
        super(AutoModel, self).__init__()

        self.model_init = model_init
        self.device = device
        self.task_initializer = task_initializer

        self.hidden_size = self.model_init.hidden_size
        self.print = printer[(self.__class__.__name__, '-', Color.MAGENTA)]

        self.model = model_class(self.model_init.model_config)  # use compatible code
        self.embedding_tables = self.model_init.get_embedding_tables()
        self.extra_modules = self.task_initializer.get_extra_modules()
        # self.print('Extra Modules', self.extra_modules)

        self.timer = None  # type: Optional[Timer]

    def set_timer(self, timer):
        self.timer = timer

    def forward(self, batch, task: Union[str, BaseTask]):
        raise NotImplementedError

from loader.dataset.bert_dataset import BertDataset
from loader.init.model_init import ModelInit
from model.network.caser import CaserConfig


class CaserInit(ModelInit):
    def __init__(
            self,
            num_vertical,
            num_horizontal,
            max_length,
            dropout=0.1,
            **kwargs
    ):
        super(CaserInit, self).__init__(**kwargs)
        self.num_vertical = num_vertical
        self.num_horizontal = num_horizontal
        self.max_length = max_length
        self.dropout = dropout

    def load_model_config(self):
        assert isinstance(self.dataset, BertDataset)

        return CaserConfig(
            hidden_size=self.hidden_size,
            num_vertical=self.num_vertical,
            num_horizontal=self.num_horizontal,
            dropout=self.dropout,
            max_length=self.max_length,
        )

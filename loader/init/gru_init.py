from loader.dataset.bert_dataset import BertDataset
from loader.init.model_init import ModelInit
from model.network.gru import GruConfig


class GruInit(ModelInit):
    def __init__(
            self,
            num_hidden_layers=12,
            dropout=0.1,
            **kwargs
    ):
        super(GruInit, self).__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

    def load_model_config(self):
        assert isinstance(self.dataset, BertDataset)

        return GruConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.dataset.max_sequence,
            dropout=self.dropout,
        )

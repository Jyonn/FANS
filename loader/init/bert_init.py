from transformers import BertConfig

from loader.dataset.bert_dataset import BertDataset
from loader.init.model_init import ModelInit


class BertInit(ModelInit):
    def __init__(
            self,
            num_hidden_layers=12,
            num_attention_heads=12,
            **kwargs
    ):
        super(BertInit, self).__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

    def load_model_config(self):
        assert isinstance(self.dataset, BertDataset)

        return BertConfig(
            vocab_size=1,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.hidden_size * 4,
            max_position_embeddings=self.dataset.max_sequence,
            type_vocab_size=self.dataset.token_types,
        )

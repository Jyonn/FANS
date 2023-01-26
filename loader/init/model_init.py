from typing import Optional

import torch
from torch import nn

from loader.dataset.model_dataset import ModelDataset
from loader.embedding_init import EmbeddingInit
from utils.smart_printer import printer


class TransformEmbedding(nn.Module):
    def __init__(self, embedding_table: nn.Embedding, from_dim: int, to_dim: int):
        super(TransformEmbedding, self).__init__()
        self.embedding_table = embedding_table
        self.linear = nn.Linear(from_dim, to_dim)

    def forward(self, indexes):
        return self.linear(self.embedding_table(indexes))


class ModelInit:
    def __init__(
            self,
            dataset: ModelDataset,
            hidden_size: int = 768,
            embedding_init: EmbeddingInit = None,
            global_freeze: bool = False,
            **kwargs,
    ):
        self.print = printer.MODEL__INIT_Cblue_
        self.dataset = dataset
        self.depot = dataset.depot
        self.hidden_size = hidden_size
        self.embedding_init = embedding_init
        self.global_freeze = global_freeze

        self._embedding_tables = None
        self._model_config = None

    def load_model_config(self):
        raise NotImplementedError

    @property
    def model_config(self):
        if self._model_config:
            return self._model_config
        self._model_config = self.load_model_config()
        return self._model_config

    def get_embedding_tables(self):
        if self._embedding_tables:
            return self._embedding_tables

        embedding_tables = dict()
        required_vocabs = set()
        for col_name in self.dataset.use_cols:
            required_vocabs.add(self.dataset.depot.get_vocab(col_name))

        self.print('set global freeze to', self.global_freeze)

        for vocab in required_vocabs:
            embedding = self.embedding_init.get_embedding(vocab)  # type: Optional[torch.Tensor]
            if embedding is not None:
                self.print.LOAD_M_(vocab, '( require_grad =', not self.embedding_init.is_freezing(vocab), '), embedding with shape', embedding.shape,
                                   'and the expected shape is', self.depot.get_vocab_size(vocab, as_vocab=True), 'x', self.hidden_size)
                assert int(embedding.shape[0]) == self.depot.get_vocab_size(vocab, as_vocab=True)
                # assert embedding.shape == (self.depot.get_vocab_size(vocab, as_vocab=True), self.hidden_size)
                embedding_tables[vocab] = nn.Embedding.from_pretrained(embedding)
                embedding_tables[vocab].weight.requires_grad = not self.embedding_init.is_freezing(vocab)

                if int(embedding.shape[1]) != self.hidden_size:
                    self.print.ALIGN_M_('transform embedding from', int(embedding.shape[1]), 'to', self.hidden_size)
                    embedding_tables[vocab] = TransformEmbedding(
                        embedding_table=embedding_tables[vocab],
                        from_dim=int(embedding.shape[1]),
                        to_dim=self.hidden_size
                    )

            else:
                self.print.CREATE_M_(vocab, '( require_grad =', not self.global_freeze, '), embedding with shape', self.depot.get_vocab_size(vocab, as_vocab=True), 'x', self.hidden_size)
                embedding_tables[vocab] = nn.Embedding(
                    num_embeddings=self.depot.get_vocab_size(vocab, as_vocab=True),
                    embedding_dim=self.hidden_size
                )
                embedding_tables[vocab].weight.requires_grad = not self.global_freeze

        self.print.CREATE_M_(self.dataset.special_id, 'embedding with shape', len(self.dataset.special_tokens), 'x', self.hidden_size)
        embedding_tables[self.dataset.special_id] = nn.Embedding(
            num_embeddings=len(self.dataset.special_tokens),
            embedding_dim=self.hidden_size
        )
        embedding_tables[self.dataset.special_id].weight.requires_grad = not self.global_freeze

        self._embedding_tables = nn.ModuleDict(embedding_tables)
        return self._embedding_tables

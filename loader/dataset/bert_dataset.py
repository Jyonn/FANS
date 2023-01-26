from loader.dataset.model_dataset import ModelDataset
from loader.dataset.order import Order


class BertDataset(ModelDataset):

    def _format_expand_tokens(self, expand_tokens):
        expand_tokens_ = []
        for token in expand_tokens or []:
            if self.COL_PH in token:
                for col in self.order:
                    expand_tokens_.append(token.replace(self.COL_PH, col.name))
            else:
                expand_tokens_.append(token)
        return expand_tokens_

    def __init__(
            self,
            order: list,
            **kwargs
    ):
        super(BertDataset, self).__init__(**kwargs)

        self.order = Order(order)
        self.use_cols = self._format_use_cols(self.order)

        self.max_sequence = self._init_max_sequence(self.order)
        self.token_types = len(self.order) if self.use_sep_token else 1

        self.init()

    def _build_format_data(self, sample):
        return self.build_format_sequence(sample, self.order)

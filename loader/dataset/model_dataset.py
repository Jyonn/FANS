import numpy as np
import torch

from loader.dataset.order import Order
from loader.dataset.unidep_dataset import UniDepDataset
from utils.smart_printer import printer, Bracket, Color


class ModelDataset(UniDepDataset):
    special_id = '__special_id'
    special_tokens: list

    injected_task: any = None
    max_sequence: int
    use_cols: list

    TOKENS: dict = {}
    COL_PH = '{col}'

    def __init__(
            self,
            append=None,
            use_sep_token=True,
            use_cls_token=True,

            inject_task=None,
            expand_tokens=None,
            **kwargs,
    ):
        super(ModelDataset, self).__init__(**kwargs)

        self.col_info = self.depot.col_info
        self.append_cols = self._format_append(append)

        self.use_sep_token = use_sep_token
        self.use_cls_token = use_cls_token

        self.expand_tokens = expand_tokens
        self.injected_task = inject_task

        self.print = printer[(self.__class__.__name__, Bracket.CLASS, Color.MAGENTA)]

    def init(self):
        expand_tokens = self._format_expand_tokens(self.expand_tokens)
        self.special_tokens = list(range(3 + len(expand_tokens)))
        self.PAD, self.CLS, self.SEP, *token_ids = self.special_tokens

        self.TOKENS = dict(PAD=self.PAD, BOS=self.CLS, SEP=self.SEP)
        for token, token_id in zip(expand_tokens, token_ids):
            self.TOKENS[token] = token_id
        self.print('tokens', self.TOKENS)

        if self.injected_task:
            self.injected_task.injector_init(self)

    """
    get raw and pack sample
    """

    def get_raw_sample(self, index):
        sample = self.depot[index]

        if self.injected_task:
            sample = self.injected_task.sample_injector(sample)

        return sample

    def pack_sample(self, index):
        sample = self.get_raw_sample(index)
        return self.build_format_data(sample)

    def get_pad_sample(self):
        return self.pack_sample(0)

    def pack_random_sample(self):
        return self.pack_sample(np.random.randint(len(self.depot)))

    def _build_format_data(self, sample):
        raise NotImplementedError

    def build_format_data(self, sample):
        data = self._build_format_data(sample)

        append_info = dict()
        for col_name in self.append_cols:
            append_info[col_name] = torch.tensor(sample[col_name])
        data['append_info'] = append_info

        return data

    # data format

    @staticmethod
    def _format_use_cols(order: Order):
        use_cols = []
        for col in order:  # type: Order.Col
            use_cols.append(col.name)
            use_cols.extend(col.attrs)
        return use_cols

    def _format_append(self, append):
        return append or []

    def _init_max_sequence(self, order: Order):
        max_sequence = int(self.use_cls_token)  # [CLS]
        for col in order:
            max_length = self.col_info[col.name].max_length or 1
            max_sequence += max_length + int(self.use_sep_token)  # [SEP]
        return max_sequence

    def pad(self, sequence: list):
        return sequence + [self.PAD] * (self.max_sequence - len(sequence))

    def _format_expand_tokens(self, expand_tokens) -> list:
        raise NotImplementedError

    @staticmethod
    def get_feature(sample, col_name):
        feat = sample[col_name]
        if isinstance(feat, np.ndarray):
            feat = feat.tolist()
        if not isinstance(feat, list):
            feat = [feat]
        return feat

    def build_format_sequence(self, sample, order: Order):
        col_mask = dict()
        input_ids = []
        token_type_ids = []

        if self.use_cls_token:
            input_ids.append(self.CLS)
            token_type_ids.append(0)
        special_mask = torch.tensor([1] * self.max_sequence, dtype=torch.long)
        attention_mask = torch.tensor([1] * self.max_sequence, dtype=torch.long)
        position = len(input_ids)
        token_type = 0

        attr_ids = dict()

        for col in order:  # type: Order.Col
            attr_col_names = col.attrs
            if col.attrs:
                attr_ids[col.name] = dict()

            feat = self.get_feature(sample, col.name)
            col_mask[col.name] = torch.tensor([0] * self.max_sequence, dtype=torch.long)
            col_mask[col.name][position: position + len(feat)] = 1
            special_mask -= col_mask[col.name]

            for attr_col_name in attr_col_names:
                attr_feat = self.get_feature(sample, attr_col_name)
                assert len(feat) == len(attr_feat)
                attr_ids[col.name][attr_col_name] = torch.tensor([
                    *([0] * position),
                    *attr_feat,
                    *([0] * (self.max_sequence - len(attr_feat) - position))
                ], dtype=torch.long)

            input_ids.extend(feat)
            position += len(feat)
            token_type_ids.extend([token_type] * len(feat))

            if self.use_sep_token:
                token_type_ids.append(token_type)
                input_ids.append(self.SEP)
                position += 1
                token_type += 1

        attention_mask[position:] = 0
        input_ids = torch.tensor(self.pad(input_ids), dtype=torch.long)
        token_type_ids = torch.tensor(self.pad(token_type_ids), dtype=torch.long)
        col_mask[self.special_id] = special_mask

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=token_type_ids,
            col_mask=col_mask,
            attr_ids=attr_ids,
        )

        # for k in d:
        #     if isinstance(d[k], torch.Tensor):
        #         if int(d[k].shape[0]) != self.max_sequence:
        #             print(k, len(d[k]), d[k])
        #             print(sample)
        #             exit(0)
        # return d

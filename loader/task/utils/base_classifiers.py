import torch
from torch import nn
from transformers import BartConfig, BertConfig
from transformers.activations import ACT2FN

from utils.smart_printer import printer, Color


class TransformLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            activation_function,
            layer_norm_eps=None,
    ):
        super(TransformLayer, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = ACT2FN[activation_function]
        if layer_norm_eps is None:
            self.LayerNorm = nn.LayerNorm(hidden_size)
        else:
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            vocab_size,
    ):
        super(DecoderLayer, self).__init__()
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states) -> torch.Tensor:
        return self.decoder(hidden_states)


class BaseClassifier(nn.Module):
    classifiers = dict()

    def __init__(
            self,
            vocab_size,
            hidden_size,
            activation_function,
            layer_norm_eps=None,
    ):
        super(BaseClassifier, self).__init__()

        self.print = printer[(self.__class__.__name__, '-', Color.YELLOW)]

        self.transform_layer = TransformLayer(
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )
        self.decoder_layer = DecoderLayer(
            hidden_size=hidden_size,
            vocab_size=vocab_size
        )

    @classmethod
    def create(cls, key, vocab_size, hidden_size, activation_function, layer_norm_eps=None):
        if key in cls.classifiers:
            return cls.classifiers[key]

        classifier = cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )
        cls.classifiers[key] = classifier
        return classifier

    def forward(self, last_hidden_states):
        hidden_states = self.transform_layer(last_hidden_states)
        prediction = self.decoder_layer(hidden_states)
        return prediction


class BartClassifier(BaseClassifier):

    @classmethod
    def create(
        cls,
        config: BartConfig,
        key,
        vocab_size,
        **kwargs
    ):
        return super().create(
            key=key,
            vocab_size=vocab_size,
            hidden_size=config.d_model,
            activation_function=config.activation_function,
        )


class BertClassifier(BaseClassifier):
    @classmethod
    def create(
            cls,
            config: BertConfig,
            key,
            vocab_size,
            **kwargs
    ):
        return super().create(
            key=key,
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            activation_function=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
        )


class ClusterClassifier(nn.Module):
    classifiers = dict()

    def __init__(
            self,
            cluster_vocabs,
            hidden_size,
            activation_function,
            layer_norm_eps=None,
    ):
        super(ClusterClassifier, self).__init__()
        self.n_clusters = len(cluster_vocabs)
        self.hidden_size = hidden_size

        self.print = printer[(self.__class__.__name__, '-', Color.YELLOW)]

        self.transform_layer = TransformLayer(
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=hidden_size,
                vocab_size=vocab_size
            ) for vocab_size in cluster_vocabs
        ])

    @classmethod
    def create(cls, key, cluster_vocabs, hidden_size, activation_function, layer_norm_eps=None):
        if key in cls.classifiers:
            return cls.classifiers[key]

        classifier = cls(
            cluster_vocabs=cluster_vocabs,
            hidden_size=hidden_size,
            activation_function=activation_function,
            layer_norm_eps=layer_norm_eps,
        )
        cls.classifiers[key] = classifier
        return classifier

    def forward(self, last_hidden_states: torch.Tensor, cluster_labels: torch.Tensor):
        """

        :param cluster_labels: torch.Tensor([batch_size, sequence_length])
        :param last_hidden_states: torch.Tensor([batch_size, sequence_length, hidden_size])
        """

        hidden_states = self.transform_layer(last_hidden_states)  # type: torch.Tensor
        predictions = []

        for i_cluster in range(self.n_clusters):
            mask = (cluster_labels == i_cluster).unsqueeze(dim=-1)
            if not mask.sum():
                predictions.append(None)
            else:
                cluster_hidden_states = torch.masked_select(hidden_states, mask).reshape(-1, self.hidden_size)  # [L, D]
                predictions.append(self.decoder_layers[i_cluster](cluster_hidden_states))  # [L, V]

        return predictions


class BartClusterClassifier(ClusterClassifier):
    @classmethod
    def create(cls, key, cluster_vocabs, config: BartConfig, **kwargs):
        return super().create(
            key=key,
            cluster_vocabs=cluster_vocabs,
            hidden_size=config.d_model,
            activation_function=config.activation_function,
        )


class BertClusterClassifier(ClusterClassifier):
    @classmethod
    def create(cls, key, cluster_vocabs, config: BertConfig, **kwargs):
        return super().create(
            key=key,
            cluster_vocabs=cluster_vocabs,
            hidden_size=config.hidden_size,
            activation_function=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
        )

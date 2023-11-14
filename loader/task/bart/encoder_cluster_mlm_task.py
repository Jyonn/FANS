from loader.dataset.bart_dataset import BartDataset
from loader.task.utils.base_classifiers import BartClassifier, BartClusterClassifier
from loader.task.utils.base_cluster_mlm_task import BaseClusterMLMTask

from loader.task.utils.base_mlm_task import MLMBartBatch

from utils.transformers_adaptor import Seq2SeqModelOutput


class EncoderClusterMLMTask(BaseClusterMLMTask):
    name = 'en-cluster-mlm'
    mask_scheme = 'E-C-MASK_{en-col}'
    mask_col_ph = '{en-col}'
    dataset: BartDataset
    cls_module = BartClassifier
    cluster_cls_module = BartClusterClassifier
    batcher = MLMBartBatch

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.col_pairs = [(self.k_cluster, self.k_local)]

    def init(self, **kwargs):
        super().init(**kwargs)

    def _rebuild_batch(self, batch: MLMBartBatch):
        self.prepare_batch(batch.encoder)

        self.random_mask(batch.encoder, self.k_cluster)

        return batch

    def produce_output(self, model_output: Seq2SeqModelOutput, batch: MLMBartBatch):
        return self._produce_output(model_output.encoder_last_hidden_state, batch=batch.encoder)

    def calculate_loss(self, batch: MLMBartBatch, output, **kwargs):
        return BaseClusterMLMTask.calculate_loss(
            self,
            batch=batch.encoder,
            output=output,
            weight=batch.encoder.weight,
            **kwargs
        )

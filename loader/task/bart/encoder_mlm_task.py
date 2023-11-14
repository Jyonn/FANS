from loader.dataset.bart_dataset import BartDataset
from loader.task.utils.base_classifiers import BartClassifier

from loader.task.utils.base_mlm_task import BaseMLMTask, MLMBartBatch

from utils.transformers_adaptor import Seq2SeqModelOutput


class EncoderMLMTask(BaseMLMTask):
    name = 'en-mlm'
    mask_scheme = 'E-MASK_{en-col}'
    mask_col_ph = '{en-col}'
    dataset: BartDataset
    cls_module = BartClassifier
    batcher = MLMBartBatch

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, **kwargs):
        super().init(**kwargs)
        self.col_order = self.get_col_order(self.dataset.encoder_order)

    def _rebuild_batch(self, batch: MLMBartBatch):
        self.prepare_batch(batch.encoder)

        for col_name in self.col_order:
            self.random_mask(batch.encoder, col_name)

        return batch

    def produce_output(self, model_output: Seq2SeqModelOutput, batch):
        return self._produce_output(model_output.encoder_last_hidden_state, batch)

    def calculate_loss(self, batch: MLMBartBatch, output, **kwargs):
        return super().calculate_loss(batch.encoder, output, **kwargs)

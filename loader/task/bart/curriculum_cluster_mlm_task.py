from loader.dataset.bart_dataset import BartDataset
from loader.task.utils.base_classifiers import BartClassifier, BartClusterClassifier
from loader.task.utils.base_cluster_mlm_task import BaseClusterMLMTask, ClusterMLMTaskLoss
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask, CurriculumMLMBartBatch

from utils.transformers_adaptor import Seq2SeqModelOutput


class CurriculumClusterMLMTask4Bart(BaseCurriculumMLMTask, BaseClusterMLMTask):
    name = 'cu-cluster-mlm-bart'
    dataset: BartDataset
    cls_module = BartClassifier
    cluster_cls_module = BartClusterClassifier
    batcher = CurriculumMLMBartBatch

    def _rebuild_batch(self, batch: CurriculumMLMBartBatch):
        self.prepare_batch(batch.encoder)
        self.prepare_batch(batch.decoder)
        batch.decoder.weight = batch.encoder.weight
        batch.decoder.mask_ratio = batch.encoder.mask_ratio

        if self.is_training:
            self.random_mask(batch.encoder, self.k_cluster)
        self.left2right_mask(batch.decoder, self.p_cluster)

        return batch

    def produce_output(self, model_output: Seq2SeqModelOutput, batch: CurriculumMLMBartBatch):
        return (
            self._produce_output(model_output.encoder_last_hidden_state, batch=batch.encoder),
            self._produce_output(model_output.last_hidden_state, batch=batch.decoder),
        )

    def calculate_loss(self, batch: CurriculumMLMBartBatch, output, **kwargs):
        encoder_loss = BaseClusterMLMTask.calculate_loss(
            self,
            batch=batch.encoder,
            output=output[0],
            weight=batch.encoder.weight,
            **kwargs
        )
        decoder_loss = BaseClusterMLMTask.calculate_loss(
            self,
            batch=batch.decoder,
            output=output[1],
            weight=batch.decoder.weight,
            **kwargs
        )
        return ClusterMLMTaskLoss(
            local_loss=encoder_loss.local_loss + decoder_loss.local_loss,
            cluster_loss=encoder_loss.cluster_loss + decoder_loss.cluster_loss,
        )

    def test__curriculum(self, batch: CurriculumMLMBartBatch, output, metric_pool):
        return BaseClusterMLMTask.test__curriculum(
            self,
            batch.decoder,
            output[1],
            metric_pool=metric_pool
        )

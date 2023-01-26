from loader.dataset.bert_dataset import BertDataset
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask, CurriculumMLMBertBatch
from loader.task.utils.base_classifiers import BertClassifier
from utils.transformers_adaptor import BertOutput


class CurriculumMLMTask(BaseCurriculumMLMTask):
    name = 'cu-mlm'
    dataset: BertDataset
    cls_module = BertClassifier
    batcher = CurriculumMLMBertBatch

    def __init__(
            self,
            known_items='known_items',
            pred_items='pred_items',
            **kwargs,
    ):
        super(CurriculumMLMTask, self).__init__(**kwargs)

        self.known_items = known_items
        self.pred_items = pred_items
        self.col_order = [self.known_items, self.pred_items]

    def _rebuild_batch(self, batch):
        self.prepare_batch(batch)

        if self.is_training:
            self.random_mask(batch, self.known_items)
        self.left2right_mask(batch, self.pred_items)

        return batch

    def produce_output(self, model_output: BertOutput, batch: CurriculumMLMBertBatch):
        return self._produce_output(model_output.last_hidden_state, batch)

    def test__curriculum(self, batch: CurriculumMLMBertBatch, output, metric_pool):
        mask_labels_col = batch.mask_labels_col
        indexes = batch.append_info['index']
        self._test__curriculum(
            indexes=indexes,
            mask_labels_col=mask_labels_col,
            output=output,
            metric_pool=metric_pool,
            col_name=self.pred_items,
        )

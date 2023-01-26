import torch

from loader.dataset.bert_dataset import BertDataset
from loader.task.utils.base_cluster_mlm_task import BaseClusterMLMTask
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask, CurriculumMLMBertBatch
from loader.task.utils.base_classifiers import BertClusterClassifier, BertClassifier

from utils.transformers_adaptor import BertOutput


class CurriculumClusterMLMTask(BaseCurriculumMLMTask, BaseClusterMLMTask):
    name = 'cu-cluster-mlm'
    dataset: BertDataset
    cls_module = BertClassifier
    cluster_cls_module = BertClusterClassifier
    batcher = CurriculumMLMBertBatch

    def _rebuild_batch(self, batch):
        self.prepare_batch(batch)

        if self.is_training:
            self.random_mask(batch, self.k_cluster)
        self.left2right_mask(batch, self.p_cluster)

        return batch

    def produce_output(self, model_output: BertOutput, batch: CurriculumMLMBertBatch):
        return self._produce_output(model_output.last_hidden_state, batch)

    def test__curriculum(self, batch: CurriculumMLMBertBatch, output, metric_pool):
        return BaseClusterMLMTask.test__curriculum(
            self,
            batch,
            output,
            metric_pool=metric_pool
        )

    def calculate_loss(self, batch: CurriculumMLMBertBatch, output, **kwargs):
        return BaseClusterMLMTask.calculate_loss(
            self,
            batch=batch,
            output=output,
            weight=batch.weight,
            **kwargs
        )

    def test__left2right(self, samples, model, metric_pool, dictifier, k):
        for sample in samples:
            ground_truth = sample[self.p_global][:]
            length = len(ground_truth)
            arg_sorts = []

            times = (length + k - 1) // k
            print(sample)
            exit(0)
            for i in range(times):
                sample[self.p_global] = ground_truth[i * k: max((i + 1) * k, length)]
                batch = dictifier([self.dataset.build_format_data(sample)])
                batch = self.rebuild_batch(batch)  # type: CurriculumMLMBertBatch

                outputs = model(
                    batch=batch,
                    task=self,
                )[self.depot.get_vocab(self.concat_col)]  # [B, S, V]

                pred_cluster_labels = outputs['pred_cluster_labels'][0]
                outputs = outputs[self.p_local]
                col_mask = batch.mask_labels_col[self.p_cluster][0]
                cluster_indexes = [0] * self.n_clusters

                for i_tok in range(self.dataset.max_sequence):
                    if col_mask[i_tok]:
                        cluster_id = pred_cluster_labels[i_tok]
                        top_items = torch.argsort(
                            outputs[cluster_id][cluster_indexes[cluster_id]], descending=True
                        ).cpu().tolist()[:metric_pool.max_n]
                        top_items = [self.local_global_maps[cluster_id][item] for item in top_items]
                        arg_sorts.append(top_items)
                        cluster_indexes[cluster_id] += 1
                    else:
                        arg_sorts.append(None)

                # sample[self.k_]
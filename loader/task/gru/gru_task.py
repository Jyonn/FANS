from loader.task.bert.bert4rec_task import Bert4RecBatch
from loader.task.bert.sasrec_task import SASRecTask


class GruTask(SASRecTask):
    name = 'gru'

    def produce_output(self, last_hidden_states, batch: Bert4RecBatch):
        return self._produce_output(last_hidden_states, batch)

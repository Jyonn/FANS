from loader.task.base_task import BaseTask


class NonTask(BaseTask):
    name = 'non'

    def _init_extra_module(self):
        return None

    def produce_output(self, model_output: any, **kwargs):
        return model_output

    def _rebuild_batch(self, batch):
        return batch

    def calculate_loss(self, batch, output, **kwargs):
        pass

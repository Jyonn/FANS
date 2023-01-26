import torch


class TaskLoss:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss

    def backward(self):
        if self.loss.requires_grad:
            self.loss.backward()

    def get_loss_dict(self) -> dict:
        return getattr(self, '__dict__')


class LossDepot:
    def __init__(self):
        self.depot = dict()

    def add(self, loss: TaskLoss):
        loss_dict = loss.get_loss_dict()

        for loss_name, loss_value in loss_dict.items():
            if loss_name not in self.depot:
                self.depot[loss_name] = []
            self.depot[loss_name].append(loss_value.detach().cpu().item())

    def summarize(self):
        for loss_name in self.depot:
            self.depot[loss_name] = torch.tensor(self.depot[loss_name]).mean().item()
        return self

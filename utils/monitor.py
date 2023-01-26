import json
import os

import torch
from oba import Obj


class Monitor:
    def __init__(
            self,
            super_store=False,
            interval=None,
            monitor=None,
            ckpt_path=None,
            task=None,
            top=None,
            epoch_skip=None,
    ):
        self.interval = interval
        self.super_store = super_store
        self.candidates = []
        self.monitor = monitor
        self.ckpt_path = ckpt_path
        self.task = task
        self.top = top
        self.epoch_skip = epoch_skip

    def remove_checkpoint(self, epoch):
        epoch_path = os.path.join(self.ckpt_path, 'epoch_{}.bin'.format(epoch))
        os.system(f'rm {epoch_path}')

    def store_checkpoint(self, epoch, state_dict):
        epoch_path = os.path.join(self.ckpt_path, 'epoch_{}.bin'.format(epoch))
        torch.save(state_dict, epoch_path)

    def push(self, epoch, loss_depots: dict, state_dict):
        if self.epoch_skip and epoch < self.epoch_skip:
            return

        if not self.super_store:
            if (epoch + 1) % self.interval == 0:
                self.store_checkpoint(epoch, state_dict)
            return

        if len(loss_depots) == 1:
            loss = list(loss_depots.values())[0]
        else:
            loss = loss_depots[self.task]
        loss = Obj(loss)
        self.candidates.append((epoch, loss))

        stay = [True] * len(self.candidates)
        for ia in range(len(self.candidates)):
            for ib in range(len(self.candidates)):
                if ia == ib or not stay[ia] or not stay[ib]:
                    continue
                a, b = self.candidates[ia][1], self.candidates[ib][1]
                if eval(self.monitor):
                    stay[ib] = False

        for i in range(len(self.candidates) - 1):
            if not stay[i]:
                self.remove_checkpoint(self.candidates[i][0])

        self.candidates = [self.candidates[i] for i in range(len(self.candidates)) if stay[i]]

        if not stay[-1]:
            return

        self.store_checkpoint(epoch, state_dict)

    def export(self):
        if self.top:
            for candidate in self.candidates[:-self.top]:
                self.remove_checkpoint(candidate[0])
            self.candidates = self.candidates[-self.top:]
        candidates = list(map(lambda x: x[0], self.candidates))
        export_path = os.path.join(self.ckpt_path, 'candidates.json')
        json.dump(candidates, open(export_path, 'w'))

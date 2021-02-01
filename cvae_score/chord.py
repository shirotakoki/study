import pandas as pd
import numpy as np
import torch
from torch import nn, optim


class chord(nn.Module):
    def __init__(self):
        super(chord, self).__init__()
        self.chord_list = ['c', 'c#', 'd', 'd#', 'e',
                           'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'maj_flg', 'min_flg', 'duration']
        self.data_idx = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'rest',
                         'c_h', 'c#_h', 'd_h', 'd#_h', 'e_h', 'f_h', 'f#_h', 'g_h', 'g#_h',
                         'a_h', 'a#_h', 'b_h', 'rest_h']
        self.note_property = {'REST_IDX': 12, 'REST_H_IDX': 25}
        """
        self.majscale_weight_table = [
            0.2, 0.0, 0.05, 0.0, 0.4, 0.05, 0.0, 0.2, 0.0, 0.05, 0.0, 0.05]
        """

        self.majscale_weight_table = torch.tensor([
            0.6, 0.0, 0.2, 0.0, 1.0, 0.2, 0.0, 0.6, 0.0, 0.2, 0.0, 0.2])
        self.majscale_weight_table_exam = torch.tensor([
            0.6, 0.0, 0.2, 0.0, 1.0, 0.2, 0.0, 0.6, 0.0, 0.2, 0.0, 0.2, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
        self.majscale_weight_dict = {'c': 0.2, 'c#': 0.0, 'd': 0.05, 'd#': 0.0, 'e': 0.4,
                                     'f': 0.05, 'f#': 0.0, 'g': 0.2, 'g#': 0.05, 'a': 0.0, 'a#': 0.0, 'b': 0.05}
        self.mask = torch.rand(16, (12 + 1) * 2)  # 損失関数テスト用

    """
    def test_calc(self, data, label):
        res = torch.zeros(1, dtype=torch.float, requires_grad=True)
        # print(data.to('cpu').detach().numpy().copy())
        # print(data[0][0])
        # print(res)
        # print(notes_data)
        for i, d_i in enumerate(data):
            duration = self.calc_duration(data, i)
            # print(duration)
            res = torch.add(
                res, self.calc_consonance(d_i, label, duration))
        res = res / torch.tensor([label[0][self.chord_list.index('duration')]])
        print(res)
        return res
    """

    def test_calc(self, data, label):
        res = torch.zeros((1, 1), dtype=torch.float, requires_grad=False)
        # print(data.to('cpu').detach().numpy().copy())
        # print(data[0][0])
        # print(res)
        # print(notes_data)
        cp_data = data.clone()
        # print(data)
        for i in range(data.size()[0]):
            duration = self.calc_duration(data, i)
            # print(duration)
            res = res + torch.tensor(self.calc_consonance(data[i], label,
                                                          duration)).clone().detach().requires_grad_(True)
        res = res / torch.tensor([label[0][self.chord_list.index('duration')]])
        # print(res)
        return res

    def loss_exam(self, data, label):
        #data = torch.sum(data)/(data.size()[0]*data.size()[1])
        #data = data*self.mask

        w = torch.zeros((data.size()[0], data.size()[1]))
        sum_dur = 0.0
        for i in range(data.size()[0]):
            duration = torch.tensor(self.calc_duration(data, i))
            w[i] = data[i] * self.majscale_weight_table_exam
            sum_dur += duration
        # print(w)
        div = w.size()[0] * w.size()[1] * sum_dur
        if div == 0:
            div = 1.0
        w = torch.sum(w) / (w.size()[0] * w.size()[1])
        return w

    def calc_consonance(self, d_i, label, duration):
        threshold = 0.7
        pitch_d_i = torch.narrow(d_i, 0, 0, 12)
        max_val = torch.max(pitch_d_i)
        if max_val < threshold:
            return 0.0
        max_pitch_idx = (pitch_d_i == torch.max(pitch_d_i)).nonzero()
        res = torch.tensor(pitch_d_i[max_pitch_idx] *
                           self.majscale_weight_table[max_pitch_idx] *
                           torch.tensor([duration])).clone().detach().requires_grad_(True)
        return res

    def calc_duration(self, data, time_idx):
        threshold = 0.7
        pitch_data = torch.narrow(data, 1, 0, 12)
        max_pitch_idx = (pitch_data[time_idx] == torch.max(
            pitch_data[time_idx])).nonzero()[0]
        # print(max_pitch_idx)
        if pitch_data[time_idx, max_pitch_idx].item() < threshold:
            return 0.0
        cnt = 1.0
        for d_i in torch.narrow(data, 0, time_idx+1, 15 - time_idx):
            # print(d_i)
            if d_i[max_pitch_idx + 13].item() > threshold:
                cnt += 1
            else:
                break
        # print("end")
        return cnt

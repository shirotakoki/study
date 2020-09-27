import pandas as pd
import numpy as np
import torch


class chord():
    def __init__(self):
        self.chord_list = ['c', 'c#', 'd', 'd#', 'e',
                           'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'maj_flg', 'min_flg', 'duration']
        self.data_idx = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'rest',
                         'c_h', 'c#_h', 'd_h', 'd#_h', 'e_h', 'f_h', 'f#_h', 'g_h', 'g#_h',
                         'a_h', 'a#_h', 'b_h', 'rest_h']
        self.note_property = {'REST_IDX': 12, 'REST_H_IDX': 25}
        self.majscale_weight_table = [
            0.2, 0.0, 0.05, 0.0, 0.4, 0.05, 0.0, 0.2, 0.0, 0.05, 0.0, 0.05]
        self.majscale_weight_dict = {'c': 0.2, 'c#': 0.0, 'd': 0.05, 'd#': 0.0, 'e': 0.4,
                                     'f': 0.05, 'f#': 0.0, 'g': 0.2, 'g#': 0.05, 'a': 0.0, 'a#': 0.0, 'b': 0.05}

    def test_calc(self, data, label):
        res = torch.tensor([[0]], dtype=torch.float)
        # print(data.to('cpu').detach().numpy().copy())
        # print(data[0][0])
        # print(res)
        # print(notes_data)
        for i, d_i in enumerate(data):
            torch.add(
                res, self.calc_consonance(d_i, label), out=res)
            # print(res)
        return res

    def calc_consonance(self, d_i, label):
        for i, cons_val in enumerate(self.majscale_weight_table):
            None

    def calc_duration(self, data, time_idx):
		threshold = 0.7
        max_pitch_idx = data[time_idx][:self.note_property['REST_IDX']+1].index(
            max(data[time_idx][:self.note_property['REST_IDX']+1]))
        if max_pitch_idx < threshold:
            return 0.0

        cnt = 0.0
        for d_i in data[time_idx:]:
            if d_i[max_pitch_idx + 13] > threshold:
				cnt += 1
			else
				break
		return cnt
				

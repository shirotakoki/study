import pandas as pd
import numpy as np
import torch


class chord():
    def __init__(self):
        self.chord_list = ['c', 'c#', 'd', 'd#', 'e',
                           'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'hold']

    def test_calc(self, data, label):
        res = torch.tensor([[0]], dtype=torch.float)
        # print(data[0][0])
        # print(res)
        for d_i in data[0]:
            torch.add(
                res, d_i[self.chord_list.index('c')].item()/16, out=res)
            torch.add(
                res, d_i[self.chord_list.index('e')].item()/16, out=res)
            torch.add(
                res, d_i[self.chord_list.index('g')].item()/16, out=res)
            # print(res)

        return res

    def test_calc_loss(self, data, label):
        res = torch.tensor([[0]], dtype=torch.float)
        # print(data[0][0])
        # print(res)
        for d_i in data:
            torch.add(
                res, d_i[self.chord_list.index('c')].item()/16, out=res)
            torch.add(
                res, d_i[self.chord_list.index('e')].item()/16, out=res)
            torch.add(
                res, d_i[self.chord_list.index('g')].item()/16, out=res)
            # print(res)

        return res

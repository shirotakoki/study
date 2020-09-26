import pandas as pd
import numpy as np
import torch
import torch.utils.data


class Dataset_generator:
    def __init__(self):
        self.quantize_val = 16
        self.bar_val = 1
        self.scale_val = 12
        self.data_val = 3
        self.score_data = pd.DataFrame(
            np.zeros((self.quantize_val*self.bar_val *
                      self.data_val, self.scale_val+1), dtype='int'),
            columns=['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'hold'])
        self.chord_data = pd.DataFrame(
            np.zeros((self.data_val, self.scale_val+2), dtype='int'),
            columns=['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'maj_flg', 'min_flg'])
        self.gen_examdata()

    def put_note(self, note, start, len, data_idx):
        self.score_data.at[data_idx*self.quantize_val + start, note] = 1
        for i in range(len-1):
            self.score_data.at[data_idx *
                               self.quantize_val + start + i + 1, 'hold'] = 1

    def put_chord(self, chord, key, data_idx):
        self.chord_data.at[data_idx, chord] = 1
        if(key == 'maj'):
            self.chord_data.at[data_idx, 'maj_flg'] = 1
        if(key == 'min'):
            self.chord_data.at[data_idx, 'min_flg'] = 1

    def gen_examdata(self):
        c_major = ['c', 'e', 'g']
        c_scale = ['c', 'd', 'e', 'f', 'g', 'a', 'b']

        for i in range(16):
            self.put_note(c_major[i % len(c_major)], i, 1, 0)
            self.put_chord('c', 'maj', 0)

        for i in range(16):
            self.put_note(c_scale[i % len(c_scale)], i, 1, 1)
            self.put_chord('c', 'maj', 1)

        for i in range(16):
            self.put_note(c_scale[np.random.randint(7)], i, 1, 2)
            self.put_chord('c', 'maj', 2)

        # print(self.score_data)
        # print(self.chord_data)


class Test_dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dg = Dataset_generator()
        self.data = self.make_data()
        self.label = self.dg.chord_data

    def __len__(self):
        return self.dg.data_val

    def __getitem__(self, index):
        out_data = self.data[index].values
        out_label = self.label.loc[index].values
        return out_data, out_label

    def make_data(self):
        k = self.dg.quantize_val
        n = self.dg.quantize_val*self.dg.bar_val*self.dg.data_val
        dfs = [self.dg.score_data.loc[i:i+k-1, :] for i in range(0, n, k)]
        for i, df_i in enumerate(dfs):
            dfs[i] = df_i.reset_index(drop=True)
        return dfs


if __name__ == "__main__":
    td = Test_dataset()
    trainloader = torch.utils.data.DataLoader(
        td, batch_size=1, shuffle=True)
    for data, label in trainloader:
        print(data)
        print(label)
        break

import pandas as pd
import numpy as np
import torch
import torch.utils.data


class Dataset_generator:
    def __init__(self):
        self.quantize_val = 16
        self.bar_val = 1
        self.scale_val = 12
        self.data_val = 100
        self.score_data = pd.DataFrame(
            np.zeros((self.quantize_val*self.bar_val *
                      self.data_val, (self.scale_val+1)*2), dtype='int'),
            columns=['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'rest',
                     'c_h', 'c#_h', 'd_h', 'd#_h', 'e_h', 'f_h', 'f#_h', 'g_h', 'g#_h',
                     'a_h', 'a#_h', 'b_h', 'rest_h'])
        self.label_data = pd.DataFrame(
            np.zeros((self.data_val, self.scale_val+3), dtype='int'),
            columns=['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#',
                     'a', 'a#', 'b', 'maj_flg', 'min_flg', 'duration'])
        self.gen_examdata()

    def put_note(self, note, start, len, data_idx):
        self.score_data.at[data_idx*self.quantize_val + start, note] = 1
        for i in range(len-1):
            self.score_data.at[data_idx *
                               self.quantize_val + start + i + 1, note + '_h'] = 1

    def put_labels(self, chord, key, data_idx, duration):
        self.label_data.at[data_idx, chord] = 1
        if(key == 'maj'):
            self.label_data.at[data_idx, 'maj_flg'] = 1
        if(key == 'min'):
            self.label_data.at[data_idx, 'min_flg'] = 1
        self.label_data.at[data_idx, 'duration'] = duration

    def gen_examdata(self):
        c_major = ['c', 'e', 'g']
        c_scale = ['c', 'd', 'e', 'f', 'g', 'a', 'b', 'rest']
        np.random.seed(0)

        """

        for i in range(16):
            self.put_note(c_major[i % len(c_major)], i, 1, 0)
            self.put_chord('c', 'maj', 0)

        for i in range(16):
            self.put_note(c_scale[i % len(c_scale)], i, 1, 1)
            self.put_chord('c', 'maj', 1)

        for i in range(16):
            self.put_note(c_scale[np.random.randint(7)], i, 1, 2)
            self.put_chord('c', 'maj', 2)
        """
        """
        for i in range(4):
            self.put_note(c_major[0], i*4, 4, 0)
        self.put_labels('c', 'maj', 0, 16)
        """
        """
        self.put_note(c_major[0], 0, 16, 0)
        self.put_labels('c', 'maj', 0, 16)
        """
        """
        for i in range(8):
            self.put_note(c_major[0], i * 2, 1, 0)
            self.put_note(c_scale[7], i*2-1, 1, 0)
        self.put_labels('c', 'maj', 0, 8)
        """
        """
        for i in range(8):
            self.put_note(c_major[0], i * 2, 1, 0)
            self.put_note(c_scale[2], i*2-1, 1, 0)
        self.put_labels('c', 'maj', 0, 16)
        """
        """
        for i in range(16):
            self.put_note(c_scale[i % 3], i, 1, 0)
        self.put_labels('c', 'maj', 0, 16)
        """
        """
        for i in range(16):
            self.put_note(c_scale[0], i, 1, 0)
        self.put_labels('c', 'maj', 0, 16)
        """
        """
        for i in range(16):
            self.put_note(c_scale[i % 7], i, 1, 0)
        self.put_labels('c', 'maj', 0, 16)
        """
        """
        for i in range(4):
            self.put_note(c_scale[i], i*4, 4, 0)
        self.put_labels('c', 'maj', 0, 16)
        """
        """
        for i in range(4):
            self.put_note(c_scale[0], i * 4, 1, 0)
            self.put_note(c_scale[1], i * 4 + 1, 1, 0)
            self.put_note(c_scale[2], i * 4 + 2, 1, 0)
            self.put_note(c_scale[1], i*4+3, 1, 0)
        self.put_labels('c', 'maj', 0, 16)
        """
        """
        for i in range(4):
            self.put_note(c_scale[0], i * 4, 1, 0)
            self.put_note(c_scale[1], i * 4 + 1, 1, 0)
            self.put_note(c_scale[2], i * 4 + 2, 1, 0)
            self.put_note(c_scale[3], i*4+3, 1, 0)
        self.put_labels('c', 'maj', 0, 16)
        """
        for i in range(self.data_val):
            for j in range(16):
                self.put_note(c_scale[np.random.randint(7)], j, 1, i)
                self.put_labels('c', 'maj', i, 16)
        # print(self.score_data)
        # print(self.label_data)


class Test_dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dg = Dataset_generator()
        self.data = self.make_data()
        self.label = self.dg.label_data

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
        td, batch_size=10, shuffle=True)
    # print(td.data)
    # print(td.label)
    for data, label in trainloader:
        print(data.size(), " : ", label.size())
        print(data)
        print(label)

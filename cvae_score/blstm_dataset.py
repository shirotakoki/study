import numpy as np
import torch
from torch.utils.data import Dataset

MELODY_CLASS_NUM = 14
REST_NUM = MELODY_CLASS_NUM - 2
HOLD_NUM = MELODY_CLASS_NUM - 1
CHORD_CLASS_NUM = 13
CHORD_TYPE_NUM = 5


class my_blstm_dataset(Dataset):

    def __init__(self, np_load_flag=False, debug=False,
                 data_name_ls=['16_melody', '16_chord', '16_chord_type'], transform=None):

        self.quantize_num = 16
        self.bar_num = 1
        self.melody_data = None
        self.chord_data = None
        self.chord_type_data = None
        self.transform = transform

        self.debug = debug
        if np_load_flag:
            save_path = './datasets/'
            self.melody_data, self.chord_data, self.chord_type_data = self.load_onehot_numpy()
            np.save(save_path + data_name_ls[0], self.melody_data)
            np.save(save_path + data_name_ls[1], self.chord_data)
            np.save(save_path + data_name_ls[2], self.chord_type_data)
        else:
            load_data_path = './datasets/'
            self.melody_data = np.load(
                load_data_path + data_name_ls[0] + '.npy')
            self.chord_data = np.load(
                load_data_path + data_name_ls[1] + '.npy')
            self.chord_type_data = np.load(
                load_data_path + data_name_ls[2] + '.npy')

        self.data_num = len(self.melody_data)

        if self.debug:
            print('in __init__')
            print('melody_data\'s shape', self.melody_data.shape)
            print('chord_data\'s shape', self.chord_data.shape)
            print('chord_type_data\'s shape', self.chord_type_data.shape)

    def load_onehot_numpy(self, sub_file_name=['onehot_train_melody', 'onehot_train_chord', 'onehot_train_chord_type']):
        dataset_path = './onehot/'
        melody = np.zeros(0)
        chord = np.zeros(0)
        chord_type = np.zeros(0)
        for i in range(self.data_num):
            melody = np.append(melody, np.load(dataset_path +
                                               sub_file_name[0] + str(i) + '.npy'))
            chord = np.append(chord, np.load(dataset_path +
                                             sub_file_name[1] + str(i) + '.npy'))
            chord_type = np.append(chord_type, np.load(dataset_path +
                                                       sub_file_name[2] + str(i) + '.npy'))
        melody = melody.reshape(-1, self.quantize_num *
                                self.bar_num, MELODY_CLASS_NUM).astype(np.int8)
        chord = chord.reshape(-1, self.quantize_num*self.bar_num,
                              CHORD_CLASS_NUM).astype(np.int8)
        chord_type = chord_type.reshape(-1, self.quantize_num *
                                        self.bar_num, CHORD_TYPE_NUM).astype(np.int8)
        if self.debug:
            print('in load_onehot_numpy')
            print('melody\'s shape', melody.shape)
            print('chord\'s shape', chord.shape)
            print('chord_type\'s shape', chord_type.shape)
        return melody, chord, chord_type

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_melody_data = torch.from_numpy(
            self.melody_data[idx].astype(np.float32)).clone()
        out_chord_data = torch.from_numpy(
            self.chord_data[idx].astype(np.float32)).clone()
        out_chord_type_data = torch.from_numpy(
            self.chord_type_data[idx].astype(np.float32)).clone()

        return out_melody_data, out_chord_data, out_chord_type_data


if __name__ == '__main__':
    dataset = my_blstm_dataset(debug=True)
    melody, chord, chord_type = dataset[0]
    print(len(dataset))
    print(melody.shape)
    print(chord.shape)
    print(chord_type.shape)

    print(melody)

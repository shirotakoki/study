import pandas as pd
import numpy as np
import torch
from torch import mul, nn, optim
import blstm_dataset
from blstm_dataset import my_blstm_dataset

MELODY_CLASS_NUM = 14
REST_NUM = MELODY_CLASS_NUM - 2
HOLD_NUM = MELODY_CLASS_NUM - 1
CHORD_CLASS_NUM = 13
CHORD_NONE_NUM = CHORD_CLASS_NUM - 1
CHORD_TYPE_NUM = 5

CHORD_TYPE_MAJ = 0
CHORD_TYPE_MIN = 1
CHORD_TYPE_DOM = 2
CHORD_TYPE_DIM = 3
CHORD_TYPE_NONE = 4


class chord(nn.Module):
    def __init__(self, debug=False):
        super(chord, self).__init__()
        self.chord_list = ['c', 'c#', 'd', 'd#', 'e',
                           'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'maj_flg', 'min_flg', 'duration']
        self.data_idx = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'rest',
                         'c_h', 'c#_h', 'd_h', 'd#_h', 'e_h', 'f_h', 'f#_h', 'g_h', 'g#_h',
                         'a_h', 'a#_h', 'b_h', 'rest_h']

        self.majscale_weight_table = torch.tensor([
            0.6, 0.0, 0.2, 0.0, 1.0, 0.2, 0.0, 0.6, 0.0, 0.2, 0.0, 0.2])
        self.minscale_weight_table = torch.tensor([
            0.6, 0.0, 0.2, 1.0, 0.0, 0.2, 0.0, 0.6, 0.2, 0.0, 0.2, 0.0])
        self.domscale_weight_table = torch.tensor([
            0.6, 0.0, 0.2, 0.0, 0.6, 0.2, 0.0, 0.2, 0.0, 0.2, 1.0, 0.0])
        self.dimscale_weight_table = torch.tensor([
            0.6, 0.0, 0.2, 0.6, 0.0, 0.2, 1.0, 0.0, 0.2, 0.0, 0.0, 0.2])
        self.none_weight_table = torch.tensor([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.debug = debug

    def calc_consonance(self, melody, chord, chord_type):

        on_note_timing_idx, on_note_pitch_idx, duration_tensor, total_duration = self.calc_duration(
            melody)

        data_num = len(on_note_timing_idx)
        consonance = torch.zeros((data_num))
        idx = 0
        while idx < data_num:
            tmp_note_idx = on_note_timing_idx[idx]
            tmp_weight_table = self.get_weight_table(
                chord[tmp_note_idx], chord_type[tmp_note_idx])
            if self.debug:
                print('TMP_MELODY : ', melody[tmp_note_idx])
                print('CALC_MUL_WEIGHT_MELODY : ',
                      torch.mul(melody[tmp_note_idx], tmp_weight_table))
            tmp_consonance = torch.sum(
                torch.mul(melody[tmp_note_idx], tmp_weight_table))
            tmp_consonance = torch.mul(tmp_consonance, duration_tensor[idx])

            if self.debug:
                print('TMP_CONSONANCE : ', tmp_consonance)
            consonance[idx] = tmp_consonance
            idx += 1

        consonance = consonance.sum()
        consonance = torch.div(consonance, total_duration)
        return consonance

    def get_weight_table(self, chord_i, chordtype_i):
        chordtype_num = self.enc_chord_type(chordtype_i)
        chord_num = self.enc_chord_num(chord_i)

        if chordtype_num == CHORD_TYPE_NONE or chord_num == CHORD_NONE_NUM:
            ret_table = torch.cat([self.none_weight_table, torch.zeros(2)])
            return ret_table

        ret_table = []
        if chordtype_num == CHORD_TYPE_MAJ:
            ret_table = self.majscale_weight_table
        elif chordtype_num == CHORD_TYPE_MIN:
            ret_table = self.minscale_weight_table
        elif chordtype_num == CHORD_TYPE_DOM:
            ret_table = self.domscale_weight_table
        elif chordtype_num == CHORD_TYPE_DIM:
            ret_table = self.dimscale_weight_table

        if self.debug:
            print('CHORD_TABLE_RAW : ', ret_table)

        ret_table = torch.roll(ret_table, chord_num, 0)
        ret_table = torch.cat([ret_table, torch.zeros(2)])

        if self.debug:
            print('CHORD_TABLE : ', ret_table)

        return ret_table

    def enc_chord_num(self, chord_i):
        ret_num = torch.argmax(chord_i).tolist()
        if self.debug:
            print('CHORD : ', chord_i)
            print('CHORD_NUM : ', ret_num)

        return ret_num

    def enc_chord_type(self, chord_type_i):
        chord_type_i = chord_type_i.int()
        ret_num = 0

        if chord_type_i[0] == 1:
            ret_num = CHORD_TYPE_MAJ
        elif chord_type_i[1] == 1:
            ret_num = CHORD_TYPE_MIN
        elif chord_type_i[2] == 1:
            ret_num = CHORD_TYPE_DOM
        elif chord_type_i[3] == 1:
            ret_num = CHORD_TYPE_DIM
        else:
            ret_num = CHORD_TYPE_NONE

        if self.debug:
            print('chord_type_i : ', chord_type_i)
            print('CHORD_TYPE_NUM : ', ret_num)
        return ret_num

    def on_note(self, data_i, thr):
        ret_flg = False
        on_note_pitch = None

        for i in range(REST_NUM):
            if data_i[i] > thr:
                ret_flg = True
                on_note_pitch = i

        if self.debug:
            print('-- at func on_note --')
            # print('data_i : ', data_i)
            print('flg : ', ret_flg)
            print('on_note_pitch : ', on_note_pitch)

        return on_note_pitch, ret_flg

    def calc_note_duration(self, data, idx, thr):
        data_num = len(data)
        idx += 1
        ret_duration = 1
        while idx < data_num:
            if data[idx][HOLD_NUM] > thr:
                idx += 1
                ret_duration += 1
            else:
                break
        return ret_duration

    def calc_duration(self, data):
        if self.debug:
            print('-- at func calc_duration --')

        threshold = 0.7
        data_num = len(data)
        idx = 0
        on_note_timing_idx = []
        on_note_pitch_idx = []
        duration_tensor = torch.zeros(0)

        while idx < data_num:
            on_note_pitch, is_on_note = self.on_note(data[idx], threshold)
            if is_on_note:
                d = self.calc_note_duration(data, idx, threshold)
                on_note_timing_idx.append(idx)
                on_note_pitch_idx.append(on_note_pitch)
                duration_tensor = torch.cat(
                    (duration_tensor, torch.Tensor([d])), 0)
                idx += d
                if self.debug:
                    print('on_note is True')
                    print('data index : ', idx - d)
                    print('duration : ', d)
            else:
                idx += 1
        total_duration = torch.sum(duration_tensor)
        return on_note_timing_idx, on_note_pitch_idx, duration_tensor, total_duration


if __name__ == '__main__':
    debug = True
    cd = chord(debug=debug)
    dataset = my_blstm_dataset()
    melody, chord_d, chord_type = dataset[0]
    on_note_timing_idx, on_note_pitch_idx, duration_tensor, total_duration = cd.calc_duration(
        melody)

    consonance = cd.calc_consonance(melody, chord_d, chord_type)

    if debug:
        print('--melody onehot--')
        print(melody)
        print('--chord onehot--')
        print(chord)
        print('on_note_timing_idx : ', on_note_timing_idx)
        print('on_note_pitch_idx : ', on_note_pitch_idx)
        print('duration_tensor : ', duration_tensor)
        print('total_duration : ', total_duration)
        print('consonance : ', consonance)

    consonance_ls = []
    cnt = 0
    cd = chord(debug=False)
    for melody, chord_d, chord_typein in dataset:
        consonance_ls.append(cd.calc_consonance(melody, chord_d, chord_type))

        if consonance_ls[-1] > 1.0 or consonance_ls[-1] < 0.0:
            print('calculation_error')
        if cnt % 1000 == 0:
            print('tmp_cnt : ', cnt, '/', len(dataset))
        cnt += 1

    if debug:
        print(consonance_ls[:10])
        print(len(consonance_ls))

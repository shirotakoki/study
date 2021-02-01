import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    drop_list = ["c_h", "c#_h", "d_h", "d#_h", "e_h", "f_h", "f#_h", "g_h", "g#_h",
                 "a_h", "a#_h", "b_h", "rest_h"]
    df_0 = pd.read_csv(
        "results_rand_2/test_out_0.csv", index_col=0).drop(drop_list, axis=1)
    df_1 = pd.read_csv(
        "results_rand_2/test_out_3.csv", index_col=0).drop(drop_list, axis=1)
    df_2 = pd.read_csv(
        "results_rand_2/test_out_6.csv", index_col=0).drop(drop_list, axis=1)
    df_3 = pd.read_csv(
        "results_rand_2/test_out_9.csv", index_col=0).drop(drop_list, axis=1)
    df_0.columns = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
                    "A", "A#", "B", "rest"]
    df_1.columns = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
                    "A", "A#", "B", "rest"]
    df_2.columns = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
                    "A", "A#", "B", "rest"]
    df_3.columns = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
                    "A", "A#", "B", "rest"]
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 16))
    col = 'Greys'
    max = 1.0
    label_size = 13
    sns.heatmap(df_0, ax=axes[0], cmap=col, vmax=max)
    sns.heatmap(df_1, ax=axes[1], cmap=col, vmax=max)
    sns.heatmap(df_2, ax=axes[2], cmap=col, vmax=max)
    sns.heatmap(df_3, ax=axes[3], cmap=col, vmax=max)
    axes[0].set_ylabel('time', fontsize=label_size)
    axes[1].set_ylabel('time', fontsize=label_size)
    axes[2].set_ylabel('time', fontsize=label_size)
    axes[3].set_ylabel('time', fontsize=label_size)
    axes[0].set_title('τ=0.1')
    axes[1].set_title('τ=0.4')
    axes[2].set_title('τ=0.7')
    axes[3].set_title('τ=1.0')
    fig.savefig('heatmap.svg')

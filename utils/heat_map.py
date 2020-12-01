import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(data, is_show=False):
    plt.figure(figsize=(15, 2))
    # SKVMN cmap=RdPu
    ax = sns.heatmap(data, cmap="RdPu")
    if is_show:
        plt.show()
    return



def main():
    batch_size = 1
    max_seq_len = 50
    output_dim = 110

    # 20 * 1
    test_q = torch.randint(1, 6, (max_seq_len, ))
    test_l = torch.randint(0, 2, (max_seq_len, ))

    unique_q = torch.unique(test_q)

    test_output = []
    for i in range(output_dim):
        test_output.append(torch.linspace(0, 1, max_seq_len).unsqueeze(0))
    test_output = torch.cat(test_output, 0).transpose(0, 1)
    print(test_output.shape)

    heat_input = test_output[:, unique_q - 1].transpose(0, 1)
    plot_heatmap(heat_input, True)


if __name__ == '__main__':
    main()
import sys
import torch
from experiment.CAKT_experiment import CAKT_main


def run_CAKT():
    H = 15
    knowledge_length = 110

    best_test_auc = CAKT_main(
        model_name="CAKT",
        env='CAKT',
        data_source="assist2009", # used dataset
        k_frames=4, # hyper-param k used in 3D convolution
        batch_size=80, # batch size
        num_layers=1, # the number of LSTM layers
        input_dim=2 * knowledge_length, # input dimension, don't modify
        H=H, # feature map size of 3D conv
        embed_dim=H*H, # embedding dimension, don't modify'
        output_dim=knowledge_length, # equals to knowledge length of each dataset, don't modify
        
        weight_decay=1e-5, # l2 regularization term
        max_epoch=30,
        plot_every_iter=5,

        vis=False,
        issave=False)
    print(best_test_auc)


def run_five_dataset():
    list_datasets = [
        ("assist2009", 110),
        ("assist2015", 100),
        ("assist2017", 102),
        ("statics", 1223),
        ("synthetic", 50)
    ]

    results = {}
    for data_source, knowledge_length in list_datasets[:]:
        print(data_source, knowledge_length)

        # run CAKT
        H = 15
        best_test_auc = main(
            model_name="CAKT",
            env='CAKT',
            data_source=data_source,
            k_frames=4,
            batch_size=80,
            num_layers=1,
            input_dim=2*knowledge_length,
            H=H,
            embed_dim=H*H,
            output_dim=knowledge_length,

            weight_decay=1e-5,
            max_epoch=30,
            plot_every_iter=5,

            vis=False,
            issave=False)
        ##########################
        results[data_source] = best_test_auc
    print(results)


def run_ablation():
    ablation_list = ["LSTM_RECENT", "FC_POOLING", "FC_REAR", "WEIGHT_SUM", "NO_EXP_DECAY"]
    H = 15
    knowledge_length = 110

    # run CAKT_ablation
    best_test_auc = CAKT_main(
        model_name="CAKT_ablation",
        env='CAKT',
        ablation="NO_EXP_DECAY",
        data_source="assist2009",
        k_frames=4,
        batch_size=80,
        num_layers=1,
        input_dim=2*knowledge_length,
        H = H,
        embed_dim = H*H,
        output_dim=knowledge_length,

        weight_decay=1e-5,
        max_epoch=30,
        plot_every_iter=5,

        vis=False,
        issave=False)
    
    print(best_test_auc)


if __name__ == '__main__':
    run_CAKT()

    # run_five_dataset()
    
    # run_ablation()

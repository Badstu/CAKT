import sys
import torch
from CKT_experiment import CKT_main


def run_one_time_CKT():
    H = 15
    knowledge_length = 110

    best_test_auc = CKT_main(
        model_name="CKT",
        env='CKT',
        data_source="assist2009",
        k_frames=4,
        batch_size=80,
        num_layers=1,
        input_dim=2 * knowledge_length,
        H = H,
        embed_dim = H*H,
        output_dim=knowledge_length,
        
        weight_decay=1e-5,
        max_epoch=30,
        # lr=0.001,
        # lr_decay=0.5,
        # decay_every_epoch=10,
        cv_times=1,
        plot_every_iter=5,

        vis=False,
        issave=False)
    print(best_test_auc)


def run_10_times_CKT():
    list_best_auc = []
    H = 15
    data_source="synthetic"
    knowledge_length = 50

    for i in range(10):
        best_test_auc = CKT_main(
            model_name="CKT",
            env='CKT',
            data_source=data_source,
            k_frames=4,
            batch_size=80,
            num_layers=1,
            input_dim=2 * knowledge_length,
            H = H,
            embed_dim = H*H,
            output_dim=knowledge_length,

            weight_decay=1e-5,
            max_epoch=30,
            # lr=0.001,
            # lr_decay=0.5,
            # decay_every_epoch=10,
            cv_times=1,
            plot_every_iter=5,

            vis=False,
            issave=False)

        list_best_auc.append(best_test_auc)
        print(list_best_auc)
    
    # np.savetxt("checkpoints/best_auc_10.csv", np.array(list_best_auc))


def run_five_dataset():
    list_datasets = [
        ("assist2009", 110),
        ("assist2015", 100),
        ("assist2017", 102),
        ("statics", 1223),
        ("synthetic", 50)
    ]

    results = {}
    for data_source, knowledge_length in list_datasets[1:]:
        print(data_source, knowledge_length)

        # run CKT_WWW
        H = 15
        best_test_auc = main(
            model_name="CKT",
            env='CKT',
            data_source=data_source,
            k_frames=4,
            batch_size=80,
            num_layers=1,
            input_dim=2*knowledge_length,
            H = H,
            embed_dim = H*H,
            output_dim=knowledge_length,

            weight_decay=1e-5,
            max_epoch=30,
            # lr=0.001,
            # lr_decay=0.5,
            # decay_every_epoch=10,
            cv_times=1,
            plot_every_iter=5,

            vis=False,
            issave=False)
        ##########################
        results[data_source] = best_test_auc
    print(results)


def run_sensi_k():
    list_datasets = [
        # ("assist2009", 110),
        # ("assist2015", 100),
        ("assist2017", 102),
        # ("synthetic", 50),
        # ("statics", 1223)
    ]
    
    params_k_frames = [2, 4, 6, 8, 10, 12, 14, 16]
    batch_size = 64
    H = 15

    results = {}
    for data_source, knowledge_length in list_datasets:
        for k in params_k_frames[5:]:
            print("==============================")
            print(k, data_source, knowledge_length)

            # run CKT_WWW
            best_test_auc = CKT_main(
                model_name="CKT",
                env='CKT',
                data_source=data_source,
                k_frames=k,
                batch_size=batch_size * torch.cuda.device_count() // 2,
                num_layers=1,
                input_dim=2*knowledge_length,
                H = H,
                embed_dim = H*H,
                output_dim=knowledge_length,

                weight_decay=1e-5,
                max_epoch=20,
                cv_times=1,
                plot_every_iter=5,

                vis=False,
                issave=False)
            ##########################
            results["k={}_{}".format(k, data_source)] = best_test_auc
            print(results)
    print(results)
    ########### end of sensitive k ############


def run_sensi_H():
    '''
    要控制H的话要同时控制三个参数：H, embed_dim = hidden_dim = H*H
    '''
    list_datasets = [
        # ("assist2009", 110),
        # ("assist2015", 100),
        # ("assist2017", 102),
        ("synthetic", 50),
        ("statics", 1223)
    ]
    
    k_frames = 4
    batch_size = 64
    params_H = [11, 13, 15, 17, 19]

    results = {}
    for data_source, knowledge_length in list_datasets:
        for H in params_H[:3]:
            print("==============================")
            print(H, data_source, knowledge_length)

            # run CKT_WWW
            best_test_auc = CKT_main(
                model_name="CKT",
                env='CKT',
                data_source=data_source,
                k_frames=k_frames,
                batch_size=batch_size,
                num_layers=1,
                input_dim=2*knowledge_length,
                H = H,
                embed_dim = H*H,
                hidden_dim = H*H,
                output_dim=knowledge_length,

                weight_decay=1e-5,
                max_epoch=20,
                cv_times=1,
                plot_every_iter=5,

                vis=False,
                issave=False)
            ##########################
            results["H={}_{}".format(H, data_source)] = best_test_auc
            print(results)
    print(results)
    ########### end of sensitive H ############


def run_sensi_b():
    list_datasets = [
        # ("assist2009", 110),
        # ("assist2015", 100),
        ("assist2017", 102),
        ("synthetic", 50),
        ("statics", 1223)
    ]
    
    k_frames = 4
    params_batch_size = [8, 16, 32, 48, 64, 80, 96]
    H = 15

    results = {}
    for data_source, knowledge_length in list_datasets[::-1]:
        for b in params_batch_size[:3]:
            print("==============================")
            print(b, data_source, knowledge_length)

            # run CKT_WWW
            best_test_auc = CKT_main(
                model_name="CKT",
                env='CKT',
                data_source=data_source,
                k_frames=k_frames,
                batch_size=b * torch.cuda.device_count(),
                num_layers=1,
                input_dim=2*knowledge_length,
                H = H,
                embed_dim = H*H,
                hidden_dim = H*H,
                output_dim=knowledge_length,

                weight_decay=1e-5,
                max_epoch=20,
                cv_times=1,
                plot_every_iter=5,

                vis=False,
                issave=False)
            ##########################
            results["b={}_{}".format(b, data_source)] = best_test_auc
            print(results)
    print(results)
    ########### end of sensitive b ############


def run_params_combination():
    list_datasets = [
        ("assist2009", 110),
        ("assist2015", 100),
        ("assist2017", 102),
        ("synthetic", 50),
        ("statics", 1223)
    ]
    
    params_k_frames = [4, 8, 16, 32]
    params_batch_size = [80, 64, 48, 32]
    H = 15

    results = {}
    for data_source, knowledge_length in list_datasets:
        for k, b in zip(params_k_frames, params_batch_size):
            # if data_source == "assist2015" and (k == 4 or k == 8 or k == 16):
            #     continue
            
            print("=================================")
            print(k, b, data_source, knowledge_length)

            # run CKT_WWW
            best_test_auc = CKT_main(
                model_name="CKT",
                env='CKT',
                data_source=data_source,
                k_frames=k,
                batch_size=b,
                num_layers=1,
                input_dim=2*knowledge_length,
                H = H,
                embed_dim = H*H,
                output_dim=knowledge_length,

                weight_decay=1e-5,
                max_epoch=30,
                cv_times=1,
                plot_every_iter=5,

                vis=False,
                issave=False)
            ##########################
            results["k={}_b={}_{}".format(k, b, data_source)] = best_test_auc
            print(results)
    print(results)
    ########### end of params combination ############


def run_ablation():
    list_datasets = [
        ("assist2009", 110),
        ("assist2015", 100),
        ("assist2017", 102),
        ("synthetic", 50),
        ("statics", 1223)
    ]
    
    k_frames = 4
    batch_size = 80 * torch.cuda.device_count() if torch.cuda.device_count() > 1 else 80
    H = 15

    # ablation_list = ["LSTM_RECENT", "FC_POOLING", "FC_REAR", "WEIGHT_SUM"]
    ablation_list = ["NO_EXP_DECAY"]

    results = {}
    for data_source, knowledge_length in list_datasets:
        for ablation_option in ablation_list:
            print("=================================")
            print(ablation_option, data_source, knowledge_length)

            # run CKT_ablation
            best_test_auc = CKT_main(
                model_name="CKT_ablation",
                env='CKT',
                ablation=ablation_option,
                data_source=data_source,
                k_frames=k_frames,
                batch_size=batch_size,
                num_layers=1,
                input_dim=2*knowledge_length,
                H = H,
                embed_dim = H*H,
                output_dim=knowledge_length,

                weight_decay=1e-5,
                max_epoch=30,
                cv_times=1,
                plot_every_iter=5,

                vis=False,
                issave=False)
            ##########################
            results["ablation={}_{}".format(ablation_option, data_source)] = best_test_auc
            print(results)
    print(results)
    ########### end of run ablation ############


if __name__ == '__main__':
    # run_one_time_CKT()

    # run_10_times_CKT()

    # run_params_combination()

    # run_sensi_k()

    # run_sensi_H()

    # run_sensi_b()

    run_ablation()

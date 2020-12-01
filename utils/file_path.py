import os
from utils.config import Config


def init_file_path(opt):
    csv_pathes = []
    times = opt.cv_times
    print(times)
    # if opt.isdev:
    #     root_path = os.path.join(os.path.abspath("./"), "dataset", "assist2009_updated")
    #     csv_pathes = ["sayhi_test.csv",
    #                   "sayhi_test.csv",
    #                   "sayhi_test.csv"]
    #     csv_pathes = [os.path.join(root_path, path) for path in csv_pathes]
    #     return csv_pathes

    if opt.data_prefix:
        project_root = opt.data_prefix
    else:
        project_root = os.path.abspath("./")
    if opt.data_source == "assist2009":
        root_path = os.path.join(project_root, "dataset", "assist2009_updated")
        csv_files = [
            # "assist2009_updated_train{}.csv".format(times),
            # "assist2009_updated_valid{}.csv".format(times),
            "assist2009_updated_train.csv",
            "assist2009_updated_test.csv",
            "assist2009_updated_test.csv"
        ]
        csv_pathes = [os.path.join(root_path, path) for path in csv_files]
    elif opt.data_source == "assist2015":
        root_path = os.path.join(project_root, "dataset", "assist2015")
        csv_files = [
            # "assist2015_train{}.csv".format(times),
            # "assist2015_valid{}.csv".format(times),
            "assist2015_train.csv",
            "assist2015_test.csv",
            "assist2015_test.csv"
        ]
        csv_pathes = [os.path.join(root_path, path) for path in csv_files]
    elif opt.data_source == "assist2017":
        root_path = os.path.join(project_root, "dataset", "assist2017", "train_test")
        csv_files = [
            "assist2017_train.csv",
            "assist2017_test.csv",
            "assist2017_test.csv"
        ]
        csv_pathes = [os.path.join(root_path, path) for path in csv_files]
    elif opt.data_source == "statics":
        root_path = os.path.join(project_root, "dataset", "STATICS")
        csv_files = [
            # "STATICS_train{}.csv".format(times),
            # "STATICS_valid{}.csv".format(times),
            "STATICS_train.csv",
            "STATICS_test.csv",
            "STATICS_test.csv"
        ]
        csv_pathes = [os.path.join(root_path, path) for path in csv_files]
    elif opt.data_source == "synthetic":
        root_path = os.path.join(project_root, "dataset", "synthetic")
        csv_files = [
            # "STATICS_train{}.csv".format(times),
            # "STATICS_valid{}.csv".format(times),
            "naive_c5_q50_s4000_v0_train.csv",
            "naive_c5_q50_s4000_v0_test.csv",
            "naive_c5_q50_s4000_v0_test.csv"
        ]
        csv_pathes = [os.path.join(root_path, path) for path in csv_files]

    return csv_pathes


if __name__ == '__main__':
    opt = Config()
    init_file_path(opt)
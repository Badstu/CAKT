## Environment
This project is developed using python 3.7，Pytorch1.4.0, CUDA 10.2 on NVIDIA Titan RTX GPU. You'd better configure the environment as this.

## Quick Start

### 1. Clone the repo:
```
git clone git@github.com:Badstu/CAKT.git
cd CAKT
```

### 2. Install dependencies:
configure python, pytorch and CUDA enviroments, and then
```
pip install -r requirements.txt
```

### 3. Dataset

You can find dataset at `dataset` folder, there are five datasets used in this project.

### 4. Quick run

You can run our CAKT model with `main.py`.
```
python main.py
```

* if you want to run our main CAKT model, you can use `run_CAKT()` function, and you can easily modify some parameters, such as `k_frames`(k), `H` and `batch_size`(b) to do some experiments, for example,
    ```
    k_frames: [4, 8, 16, 32]
    H: [11, 13, 15, 17, 19]
    batch_size: [32, 48, 64, 80]
    ```

* if you want to run our model on different datasets, you can use `run_five_dataset()` function, we provide five benchmark datasets as follows:

    ```
    dataset_name, knowledge_length
    "assist2009", 110,
    "assist2015", 100,
    "assist2017", 102,
    "statics", 1223,
    "synthetic", 50
    ```
* if you want to run ablation model of CAKT, we provide `run_ablation()` function, you can set `model_name="CAKT_ablation"` and set `ablation` equal to ablation mode:
    ```
    ablation_mode: ["LSTM_RECENT", "FC_POOLING", "FC_REAR", "WEIGHT_SUM", "NO_EXP_DECAY"]
    ```

Please feel free to contact me by email to me or just leave a issue if you have any question.

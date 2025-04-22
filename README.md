# Deploying Large AI Models on Resource-Limited Devices with Split Federated Learning

SFLAM (Split Federated Learning with Attention Mechanism) is a framework based on split federated learning that divides the ViT model into client and server parts, training them collaboratively. 

### Key Features

- Support for Split Federated Learning (SFL) architecture
- Utilization of pre-trained Vision Transformer models
- Support for non-IID data distribution (controlled via Dirichlet distribution)
- Support for CIFAR-10 and CIFAR-100 datasets
- Configurable number of clients and proportion of participating clients

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- timm
- numpy
- matplotlib
- pandas

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/SFLAM.git
cd SFLAM
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the following command to start training:

```bash
python SFL_main.py
```

This will train the model on the CIFAR-10 dataset using default parameters.

### Parameter Configuration

You can customize the training process using command-line arguments:

```bash
python SFL_main.py --dataset_name cifar100 --alpha 0.1 --num_clients 100 --num_selected_clients 20 --Rounds 200 --local_epoch 10
```

#### Important Parameters

- `--dataset_name`: Dataset name, options are 'cifar10' or 'cifar100', default is 'cifar10'
- `--alpha`: Dirichlet distribution α parameter, controls non-IID degree, default is 0.5 (smaller α leads to more uneven data distribution)
- `--num_clients`: Total number of clients, default is 50
- `--num_selected_clients`: Number of clients selected for each training round, default is 10
- `--model_name`: ViT model name used (from timm library), default is 'vit_base_patch32_224'
- `--pretrained`: Whether to use pre-trained weights, default is True
- `--batch_size`: Batch size, default is 128
- `--Rounds`: Total number of federated learning rounds, default is 100
- `--local_epoch`: Number of local training epochs per round, default is 5
- `--lr`: Learning rate, default is 0.01
- `--input_size`: Input image size, default is 224 (224x224)
- `--base_dir`: Data storage directory, default is "./data"

## Project Structure

- `SFL_main.py`: Main program
- `args.py`: Parameter definitions
- `dataset.py`: Dataset processing
- `SLmodels/`
  - `FedAvg.py`: Federated Averaging algorithm implementation
  - `sflmodels.py`: Split learning model definitions
- `data/`: Data storage directory

## Citation

If you use this project, please cite the following paper:

```
@article{your-reference,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2023}
}
```

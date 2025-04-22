import numpy as np
from torchvision.datasets import CIFAR10,CIFAR100
from torchvision import transforms
import torch
import random
seed = 105
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def create_non_iid_data_distribution(alpha, num_users, dataset_name, dataset, min_data_size=400, max_data_size=500):
    if dataset_name == "cifar10":
        num_classes = 10
        targets = dataset.targets
    elif dataset_name == "cifar100":
        num_classes = 100
        targets = dataset.targets
    else:
        raise ValueError("Unsupported dataset_name")

    data_by_class = [[] for _ in range(num_classes)]
    for idx, target in enumerate(targets):
        data_by_class[target].append(idx)

    total_data_sizes = np.random.randint(min_data_size, max_data_size + 1, size=num_users)

    user_data_indices = [[] for _ in range(num_users)]

    for c in range(num_classes):
        class_data = data_by_class[c]
        if len(class_data) == 0:
            continue

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (proportions / proportions.sum())
        proportions = (proportions * len(class_data)).astype(int)  
        while sum(proportions) > len(class_data):
            proportions[np.argmax(proportions)] -= 1
        while sum(proportions) < len(class_data):
            proportions[np.argmin(proportions)] += 1

        split_indices = np.split(np.array(class_data), np.cumsum(proportions)[:-1])
        for i, split in enumerate(split_indices):
            user_data_indices[i].extend(split)

    final_user_data_indices = []
    for i in range(num_users):
        user_data = user_data_indices[i]
        target_size = total_data_sizes[i]
        if len(user_data) > target_size:
            user_data = np.random.choice(user_data, size=target_size, replace=False).tolist()
        elif len(user_data) < target_size:
            extra_samples = np.random.choice(user_data, size=target_size - len(user_data), replace=True)
            user_data = user_data + extra_samples.tolist()
        final_user_data_indices.append(user_data)

    user_class_counts = np.zeros((num_users, num_classes), dtype=int)
    for user_idx, indices in enumerate(final_user_data_indices):
        for idx in indices:
            class_idx = targets[idx]
            user_class_counts[user_idx, class_idx] += 1

    print("每个用户每个类别的数据量:")
    print(user_class_counts)
    return final_user_data_indices


def Cifar10Set(input_size = 224, batch_size = 32, base_dir = './data', num_workers = 8):
    normMean = [0.4914, 0.4822, 0.4465]
    normStd = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((input_size, input_size), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])

    training_set = CIFAR10(root=base_dir, train=True, download=True, transform=train_transform)
    validation_set = CIFAR10(root=base_dir, train=False, download=True, transform=val_transform)

    return training_set, validation_set


def Cifar100Set(input_size = 224, batch_size = 32, base_dir = './data', num_workers = 8):
    normMean = [0.4914, 0.4822, 0.4465]
    normStd = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((input_size, input_size), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])

    training_set = CIFAR100(root=base_dir, train=True, download=True, transform=train_transform)
    validation_set = CIFAR100(root=base_dir, train=False, download=True, transform=val_transform)

    return training_set, validation_set


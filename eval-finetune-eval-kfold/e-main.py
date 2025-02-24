import torch
import sys
import yaml
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import SubsetRandomSampler
import numpy as np
import os
from torch.utils.data.dataloader import DataLoader
torch.manual_seed(3407)
sys.path.append('../')
from models.resnet_base_network import ResNet18
from dataloading import CustomDataset, TensorTransforms_simple, get_small
from eval import Eval
from finetune import Finetune
from models.densenet_base_network import DenseNetModel


def loading(config, fold, skf):
    batch_size = int(config['trainer']['batch_size'])
    dataset_name = config['trainer']['name']
    small_size = get_small('../dataset/Univariate_ts_tensor/' + dataset_name + "/train",
                           '../dataset/Univariate_ts_tensor/' + dataset_name + "/test")
    train_dataset = CustomDataset('../dataset/Univariate_ts_tensor/' + config["trainer"]['name'] + "/train",
                                  '../dataset/Univariate_ts_tensor/' + config["trainer"]['name'] + "/test",
                                  transform=TensorTransforms_simple(), small_size=small_size)
    labels = train_dataset.all_labels()

    skf_indices = list(skf.split(np.zeros(len(labels)), labels))[fold]
    train_val_indices, test_indices = skf_indices
    batch_size = min(batch_size, len(test_indices))


    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.25,
                                                  stratify=np.array(labels)[train_val_indices], random_state=0)

    # label_to_indices = {label: np.where(np.array(labels) == label)[0] for label in set(labels)}
    #
    # train_indices, val_indices, test_indices = [], [], []
    #
    # for label, label_indices in label_to_indices.items():
    #     num_samples = len(label_indices)
    #     train_size = int(0.6 * num_samples)
    #     val_size = int(0.2 * num_samples)
    #
    #     np.random.seed(0)
    #     np.random.shuffle(label_indices)
    #     train_indices.extend(label_indices[:train_size])
    #     val_indices.extend(label_indices[train_size:train_size + val_size])
    #     test_indices.extend(label_indices[train_size + val_size:])

    # 创建DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=32, drop_last=False)
    val_loader = DataLoader(train_dataset, batch_size=batch_size,
                            sampler=SubsetRandomSampler(val_indices),
                            num_workers=32, drop_last=False)
    test_loader = DataLoader(train_dataset, batch_size=batch_size,
                             sampler=SubsetRandomSampler(test_indices),
                             num_workers=32, drop_last=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for folder_name in os.listdir('../runs/UCR-finetune'):
        check_file = os.path.join('../runs/UCR-finetune', folder_name, 'checkpoints', '5-fold.txt')
        if os.path.exists(check_file):
            print(f'Skipping {folder_name} as it already has 5-fold.txt')
            continue
        print(f'Running {folder_name}...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        folder_path = os.path.join('../runs/UCR-finetune', folder_name)
        config = yaml.load(open(folder_path + "/checkpoints/config.yaml", "r"), Loader=yaml.FullLoader)
        acc_sum = 0.0
        for fold in range(5):
            print(f'Running fold {fold + 1} for {folder_name}...')
            train_loader, val_loader, test_loader = loading(config, fold, skf)
            channel = config['network']['channel']

            # eval original model
            encoder=DenseNetModel(channel, **config['network'])
            #encoder = ResNet18(channel, **config['network'])
            output_feature_dim = encoder.projetion.net[0].in_features
            load_params = torch.load(
                os.path.join(folder_path + '/checkpoints/best_model.pth'),
                map_location=torch.device(torch.device(device)))
            if 'online_network_state_dict' in load_params:
                encoder.load_state_dict(load_params['online_network_state_dict'])
                print("Parameters successfully loaded.")
            encoder = encoder.to(device)
            #encoder_features = torch.nn.Sequential(*list(encoder.children())[:-1])
            encoder_features=encoder
            Eval(train_loader, test_loader, folder_path, encoder_features, 'original', fold)

            # finetune
            Finetune(train_loader, val_loader, folder_path, device, encoder, int(config['trainer']['max_epochs'] // 2))

            # eval finetune model
            encoder = torch.load(os.path.join(folder_path + '/checkpoints/best_model_fine.pth'))
            #encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
            encoder = encoder.to(device)
            f_acc = Eval(train_loader, test_loader, folder_path, encoder, 'finetune', fold)
            acc_sum = acc_sum + float(f_acc)
        print(f'Average accuracy: {acc_sum / 5}')
        with open(check_file, 'w') as file:
            file.write(str(acc_sum / 5))

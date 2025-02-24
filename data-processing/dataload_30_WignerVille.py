import os
from sklearn.preprocessing import MinMaxScaler
import tftb
import torch
from tqdm import tqdm
from aeon.datasets import load_classification
import numpy as np

Multivariate_ts_folder = "data/Multivariate_ts/"

fold_choose = ['Heartbeat', 'FingerMovements', 'HandMovementDirection', 'AtrialFibrillation', 'StandWalkJump']
from aeon.datasets import load_from_tsfile

for folder_name in os.listdir(Multivariate_ts_folder):
    if folder_name not in fold_choose:
        continue
    print(folder_name)
    # if folder_name != 'ECG200':
    #     continue

    folder_path = os.path.join(Multivariate_ts_folder, folder_name)
    save_path_train = os.path.join("data/Multivariate_ts_tensor_WVD/", folder_name, "train")
    save_path_test = os.path.join("data/Multivariate_ts_tensor_WVD/", folder_name, "test")

    processed_train_files = os.listdir(save_path_train) if os.path.isdir(save_path_train) else []
    processed_test_files = os.listdir(save_path_test) if os.path.isdir(save_path_test) else []
    X_train, y_train = load_from_tsfile(folder_path + "/" + folder_name + "_TRAIN.ts")
    X_test, y_test = load_from_tsfile(folder_path + "/" + folder_name + "_TEST.ts")
    try:
        X = np.concatenate((X_train, X_test), axis=0)
        print("Arrays can be concatenated successfully.")
    except ValueError:
        print("Arrays cannot be concatenated due to shape mismatch.")
        continue
    y = np.concatenate((y_train, y_test), axis=0)
    _, y = np.unique(y, return_inverse=True)
    if len(processed_train_files) == len(X):
        print(f"Skipping already processed folder: {folder_name}")
        continue
    print(f"Processing folder: {folder_name}")

    if not os.path.isdir(save_path_train):
        os.makedirs(save_path_train)
    if not os.path.isdir(save_path_test):
        os.makedirs(save_path_test)
    for i in tqdm(range(len(X))):
        tensor_list = []
        Label = y[i]
        for j in range(len(X[i])):
            sig = X[i][j]
            WVD = tftb.processing.WignerVilleDistribution(sig)
            WVD.run()
            WVD = WVD.tfr
            WVD = torch.tensor(WVD).unsqueeze(0)
            tensor_list.append(WVD)
        torch.save(torch.cat(tensor_list), os.path.join(save_path_train, f"{Label}_{i}_WVD.pt"))

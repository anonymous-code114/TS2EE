import os

import tftb
import torch
from tqdm import tqdm
from aeon.datasets import load_classification, load_from_tsfile
import numpy as np

univariate_ts_folder = "data/Univariate_ts/"

# fold_choose = ['FordB', 'FordA', 'ECG5000', 'Earthquakes', 'ECG200']
fold_choose = ['FordB', "FordA", 'ECG5000', 'Earthquakes', 'ECG200']

for folder_name in os.listdir(univariate_ts_folder):
    print(folder_name)
    if folder_name in fold_choose:
        continue

    folder_path = os.path.join(univariate_ts_folder, folder_name)
    save_path_train = os.path.join("data/Univariate_ts_tensor_SPWVD/", folder_name, "train")
    save_path_test = os.path.join("data/Univariate_ts_tensor_SPWVD/", folder_name, "test")

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
    y_typ, y = np.unique(y, return_inverse=True)
    if len(y_typ) == 2:
        print("no, Binary classification.")
        continue

    if len(processed_train_files) == len(X):
        print(f"Skipping already processed folder: {folder_name}")
        continue
    print(f"Processing folder: {folder_name}")

    if not os.path.isdir(save_path_train):
        os.makedirs(save_path_train)
    if not os.path.isdir(save_path_test):
        os.makedirs(save_path_test)
    flag = 0
    for i in tqdm(range(len(X))):
        Label = y[i]
        for j in range(len(X[i])):
            sig = X[i][j]
            WVD = tftb.processing.smoothed_pseudo_wigner_ville(sig)
            # WVD = tftb.processing.WignerVilleDistribution(sig)
            # WVD.run()
            # WVD=WVD.tfr
            # WVD = torch.tensor(WVD).unsqueeze(0)
        else:
            torch.save(WVD, os.path.join(save_path_train, f"{Label}_{i}_SPWVD.pt"))

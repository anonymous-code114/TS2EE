import os
import shutil

import yaml
import subprocess

import torch
torch.manual_seed(3407)

def update_yaml(file_path, new_name, new_channel=None, files_num=1024, batch_size=64):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    config['trainer']['batch_size'] = batch_size
    config['trainer']['name'] = new_name
    config['trainer']['max_epochs'] = min(max(150, files_num // int(config['trainer']['batch_size']) * 5), 240)
    print('the max_epochs is:', config['trainer']['max_epochs'])
    if new_channel:
        config['network']['channel'] = new_channel

    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)


def check_loss_file(folder_name, epochs):
    runs_folder_path = 'runs/UCR'
    for run_folder in os.listdir(runs_folder_path):
        if folder_name == run_folder.split('2024')[0]:
            loss_file_path = os.path.join(runs_folder_path, run_folder, 'checkpoints', 'loss.txt')
            if os.path.exists(loss_file_path):
                with open(loss_file_path, 'r') as file:
                    lines = file.readlines()
                    print(f'Found {len(lines)} lines in {loss_file_path}')
                    if len(lines) == int(epochs):
                        return True
                    else:
                        # shutil.rmtree(os.path.join(runs_folder_path, run_folder))
                        return False
    return False


def check_model_file(folder_name):
    runs_folder_path = 'runs/UCR'
    for run_folder in os.listdir(runs_folder_path):
        if folder_name == run_folder.split('2024')[0]:
            model_file_path = os.path.join(runs_folder_path, run_folder, 'checkpoints', 'final_model.pth')
            if os.path.exists(model_file_path):
                return True
            else:
                return False
    return False


def run_main():
    yaml_file_path = 'config/config.yaml'
    main_py_path = 'main.py'
    config = yaml.load(open(yaml_file_path, 'r'), Loader=yaml.FullLoader)
    folder_skip = ['Heartbeat', 'PhonemeSpectra', 'HandMovementDirection', 'SelfRegulationSCP1', 'UWaveGestureLibrary',
                   'ArticularyWordRecognition', 'OliveOil', 'Lightning7']

    for folder_name in os.listdir('dataset/Univariate_ts_tensor'):
        folder_path = os.path.join('dataset/Univariate_ts_tensor', folder_name)
        #if folder_name in folder_skip:
            #continue
        if os.path.isdir(folder_path):
            train_folder_path = os.path.join(folder_path, "train")
            files = os.listdir(train_folder_path)
            files_num = len(files)
            if check_model_file(folder_name):
                print(f'Skipping {folder_name} as it already has a final_model.pth')
                continue
            # if check_loss_file(folder_name, max(150, files_num // int(config['trainer']['batch_size']) * 5)):
            #     print(f'Skipping {folder_name} as it already has {config["trainer"]["max_epochs"]} lines in loss.txt')
            #     continue
            print(f'Running {folder_name}...')
            pt_file = next((file for file in files if file.endswith('.pt')), None)
            if pt_file:
                file_path = os.path.join(train_folder_path, pt_file)
                tensor = torch.load(file_path)
                channel = tensor.size(0)
            batch_size = 64
            update_yaml(yaml_file_path, folder_name, channel, files_num, batch_size)
            while True:
                try:
                    subprocess.run(['python', main_py_path], check=True)
                    break  # Exit the loop if the script runs successfully
                except subprocess.CalledProcessError:
                    batch_size = batch_size - 4
                    update_yaml(yaml_file_path, folder_name, channel, files_num, batch_size)
                    print("Error executing main.py. Retrying...")
                    print(f'batch_size: {batch_size}')

run_main()

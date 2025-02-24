import os
import re


def extract_accuracy(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        accuracy_match = re.search(r'Accuracy: (\d+\.\d+)', content)
        if accuracy_match:
            return float(accuracy_match.group(1))
    return None


def main():
    ucr_folder_path = '/home/dusa/TFA-BYOL/runs/UCR-finetune/'
    results = {}
    for subdir, dirs, files in os.walk(ucr_folder_path):
        dataset_name = subdir.split('/')[-2]
        extracted_name = dataset_name.split('2024')[0]
        if extracted_name not in results:
            results[extracted_name] = {'finetune': None, 'original': None}
        for file in files:
            if file.endswith('evaluation_results_finetune_linear.txt'):
                file_path = os.path.join(subdir, file)
                accuracy = extract_accuracy(file_path)
                if accuracy is not None:
                    print(f'Folder: {extracted_name},finetune Test Accuracy: {accuracy}')
                    results[extracted_name]['finetune'] = accuracy
            if file.endswith('evaluation_results_original_linear.txt'):
                file_path = os.path.join(subdir, file)
                accuracy = extract_accuracy(file_path)
                if accuracy is not None:
                    print(f'Folder: {extracted_name},original Test Accuracy: {accuracy}')
                    results[extracted_name]['original'] = accuracy
    for name in sorted(results):
        print(
            f'Folder: {name}, Original Test Accuracy: {results[name]["original"]}, Finetune Test Accuracy: {results[name]["finetune"]}')


if __name__ == '__main__':
    main()

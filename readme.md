# TS2EE: An Energy Contrastive Learning Model for Time Series Classification

## Usage

### 1. Data Download
Download the datasets from the following links:
- Univariate Time Series (UCR): [Download](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip)
- Multivariate Time Series (UEA): [Download](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip)

Place the downloaded files in the corresponding directories:
- `data-processing/data/Univariate_ts`
- `data-processing/data/Multivariate_ts`

### 2. Data Processing
Process the data using the provided scripts:
- For univariate time series: `data-processing/dataload_128_WignerVille.py`
- For multivariate time series: `data-processing/dataload_30_WignerVille.py`

The processed data will be stored in the following directories:
- `data-processing/data/Univariate_ts_tensor_SPWVD`
- `data-processing/data/Multivariate_ts_tensor_WVD`

### 3. Training
Move the processed data files to the dataset directory:
- `dataset/Univariate_ts_tensor`

Start the training process using the `multi-run.py` script. The trained model will be saved in the `runs/UCR` directory.

### 4. Fine-tuning and Evaluation
Move the trained model file to the `runs/UCR-finetune` directory. Start the fine-tuning and evaluation process using the `eval-finetune-eval-kfold/e-main.py` script.

## Notes
- Ensure that all necessary dependencies and libraries are installed before running the scripts.
- Adjust the file paths in the scripts as needed to match your directory structure.
- For detailed information on each step, refer to the corresponding script files.
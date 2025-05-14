# Ransomware Detection using Per-Command Labeled NVMe Streams

## Introduction

This repository is based on the paper "CLEAR: Command Level Annotated Dataset for Ransomware Detection" and provides a Python implementation to reconstruct its results and execute other functionalities.

## Installation

To run the code, follow these steps:

1. Install the required modules by running `pip install -r requirements.txt` in your terminal.
2. Ensure you have Python installed on your system.

## Running the Code

To start the application, one must do the following:
1. Configure `config.yaml` accordingly. Specifically:
1a. Set `demo_mode` to `True` to run a simplified version of the code. Setting it to `False` performs a full execution, with the parameters described below. The entire dataset should be downloaded to the `/CLEAR_Dataset` folder.
1b. Set `operational_execution_mode` to `Regular` for the regular-split experiment and `Robustness` for different robustness experiments.
1c. On `Robustness` mode, set `robustness_fold` to be either 1, 2, or 3 and `robustness_test_mode` to either `id` for an in-distribution test or `od` for an out-of-distribution test.
1d. Set the `model_name` parameter to choose the desired model. The supported models are CLT (Command-Level Transformer), PLT (Patch-Level Transformer), RF (Random Forest), DeftPunk (XGBoost-based pipeline), CommandLevelLSTM, CommandLevelLSTMContinuous, and UNet.
1e. Verify the chosen model's architecture parameters.
2. Navigate to the project directory and run `python main.py`

The code returns a pandas DataFrame of results: miss detection rate (MDR), area under the ROC curve (AUC), and Megabytes to detect (MBD) - mean and standard deviation.

### Additional Standalone Utilities
Although already present in the associated dataset, the code also supplies two utility functions (can be found under Standalone_Utils/), to allow the user to verify and/or calculate on their own NVMe data stream:
- `command_labeler.py`: Labels the dataset per command.
- `auxiliary_features_calculation.py`: Calculate the auxiliary features (WAR, RAW, etc.)

# License
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

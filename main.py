import pandas as pd
import yaml
from Misc import Utils
import os
CUR_DIR = os.getcwd()

DATA_PATH = os.path.join(CUR_DIR, 'CLEAR_Dataset')
DEMO_DATA_PATH = os.path.join(CUR_DIR, 'CLEAR_Dataset_Demo')

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def execute_experiment():
    #  Read and adapt data
    data_list = Utils.read_data(DEMO_DATA_PATH, config)  # Change to DATA_PATH for full execution

    #  Create chunks
    data_with_chunks_list = Utils.run_chunker(data_list, config)

    #  Add volume tags
    volume_chunks_list = Utils.run_volume_tagger(data_with_chunks_list)

    #  Preprocess according to the model
    if config['demo_mode']:
        volume_chunks_list = Utils.extract_train_and_test_for_demo_mode(volume_chunks_list)
    data_with_features_list = Utils.run_preprocessing(volume_chunks_list, config)

    #  Train and Infer
    chunk_results = Utils.train_and_infer(data_with_features_list, config)

    #  Evaluate
    df_results = Utils.evaluate_results(chunk_results, config)  #  MDR, AUC, MBD_Q (mean + std)

    return df_results



if __name__ == '__main__':
    df_results = execute_experiment()
    print("Experiment Finished!")

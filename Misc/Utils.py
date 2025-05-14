import os
import pandas as pd
from typing import List, Tuple, Dict, Any

from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

from Misc.ParsedData import ParsedData
from Misc.ModelInput import ModelInput
from Chunkers import ChunkByVolume, ChunkByCommands, VolumeChunkTagger
from Preprocessing import CLTTokenizer, PLTTokenizer, RFFeaturizer, DeftPunkFeaturizer
from Engines.TorchManager import TorchManager
from Engines.TreeManager import TreeManager
import numpy as np
from Misc.Constants import *
from sklearn.model_selection import GroupShuffleSplit
import scipy.stats as stats
import math
import random


def read_data(data_path, config) -> List[Any]:
    if config['demo_mode']:
        with open(os.path.join(r'Whitelists', 'Demo_Mode', 'train_demo.txt'), "r") as file:
            whitelist_train = {line.strip() for line in file}
        with open(os.path.join(r'Whitelists', 'Demo_Mode', 'test_demo.txt'), "r") as file:
            whitelist_test = {line.strip() for line in file}
    elif config['operational_execution_mode'] == 'Regular':
        with open(os.path.join(r'Whitelists', 'Regular_Split', 'train.txt'), "r") as file:
            whitelist_train = {line.strip() for line in file}
        with open(os.path.join(r'Whitelists', 'Regular_Split', 'test.txt'), "r") as file:
            whitelist_test = {line.strip() for line in file}
    elif config['operational_execution_mode'] == 'Robustness':
        if config['robustness_fold'] == 1:
            if config['robustness_test_mode'].lower() == 'id':
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_1', 'train.txt'), "r") as file:
                    whitelist_train = {line.strip() for line in file}
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_1', 'test_id.txt'), "r") as file:
                    whitelist_test = {line.strip() for line in file}
            else:  # Default is ood
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_1', 'train.txt'), "r") as file:
                    whitelist_train = {line.strip() for line in file}
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_1', 'test_ood.txt'), "r") as file:
                    whitelist_test = {line.strip() for line in file}
        elif config['robustness_fold'] == 2:
            if config['robustness_test_mode'].lower() == 'id':
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_2', 'train.txt'), "r") as file:
                    whitelist_train = {line.strip() for line in file}
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_2', 'test_id.txt'), "r") as file:
                    whitelist_test = {line.strip() for line in file}
            else:  # Default is ood
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_2', 'train.txt'), "r") as file:
                    whitelist_train = {line.strip() for line in file}
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_2', 'test_ood.txt'), "r") as file:
                    whitelist_test = {line.strip() for line in file}
        else:  # Default is 3
            if config['robustness_test_mode'].lower() == 'id':
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_3', 'train.txt'), "r") as file:
                    whitelist_train = {line.strip() for line in file}
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_3', 'test_id.txt'), "r") as file:
                    whitelist_test = {line.strip() for line in file}
            else:  # Default is ood
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_3', 'train.txt'), "r") as file:
                    whitelist_train = {line.strip() for line in file}
                with open(os.path.join(r'Whitelists', 'Robustness', 'Fold_3', 'test_ood.txt'), "r") as file:
                    whitelist_test = {line.strip() for line in file}
    else:
        print("Erroneous Execution Mode!")
        return []
    parsed_data_list = []

    #  RW
    rw_path = os.path.join(data_path, 'Ransomware')

    for rw_folder in os.listdir(rw_path):
        current_rec_path = os.path.join(rw_path, rw_folder, f'recording_{rw_folder}.parquet')
        current_data = pd.read_parquet(current_rec_path)
        adapted_current_data = adapt_data(current_data)
        train_test = "Test" if rw_folder in whitelist_test else "Train"
        if config['model_name'] == 'CLT' or config['model_name'] == 'CommandLevelLSTM' or config['model_name'] == 'CommandLevelLSTMContinuous':
            rec_max_slba = adapted_current_data.SLBA.max()
            current_parsed_data = ParsedData(1, adapted_current_data, train_test,
                                             rec_number=int(rw_folder),
                                             metadata=rec_max_slba)  # Per recording label
        else:
            current_parsed_data = ParsedData(1, adapted_current_data, train_test,
                                             rec_number=int(rw_folder))  # Per recording level
        parsed_data_list.append(current_parsed_data)

    #  Benign
    benign_path = os.path.join(data_path, 'Benign')

    for benign_folder in os.listdir(benign_path):
        current_rec_path = os.path.join(benign_path, benign_folder, f'recording_{benign_folder}.parquet')
        current_data = pd.read_parquet(current_rec_path)
        adapted_current_data = adapt_data(current_data)
        train_test = "Test" if benign_folder in whitelist_test else "Train"
        if config['model_name'] == 'CLT' or config['model_name'] == 'CommandLevelLSTM' or config['model_name'] == 'CommandLevelLSTMContinuous':
            rec_max_slba = adapted_current_data.SLBA.max()
            current_parsed_data = ParsedData(0, adapted_current_data, train_test,
                                             rec_number=int(benign_folder),
                                             metadata=rec_max_slba)  # Per recording label
        else:
            current_parsed_data = ParsedData(0, adapted_current_data, train_test,
                                             rec_number=int(benign_folder))  # Per recording level

        parsed_data_list.append(current_parsed_data)

    return parsed_data_list


def adapt_data(df_recording):
    df_recording_adapted = df_recording.rename(columns={"Offset": "SLBA", "Size": "NLB"})

    df_recording_adapted["SLBA"] //= 512
    df_recording_adapted["NLB"] //= 512
    df_recording_adapted["WAR"] //= 512
    df_recording_adapted["RAR"] //= 512
    df_recording_adapted["RAW"] //= 512
    df_recording_adapted["WAW"] //= 512

    return df_recording_adapted


def run_chunker(parsed_data_list: List[ParsedData], config):
    data_with_chunks_list = []

    if config['model_name'] == 'CLT' or config['model_name'] == 'CommandLevelLSTM' or config['model_name'] == 'CommandLevelLSTMContinuous':
        chunker = ChunkByCommands.ChunkByCommands(parsed_data_list, config)

    else:
        chunker = ChunkByVolume.ChunkByVolume(parsed_data_list)

    for parsed_data in parsed_data_list:
        parsed_df = parsed_data.data

        chunks_indices: pd.Series = chunker.create_chunk_indices_series(parsed_df)

        # Check if the indices of the Series and DataFrame are the same
        if not parsed_df.index.equals(chunks_indices.index):
            raise ValueError("Index mismatch: DataFrame and Series indices do not align")

        parsed_df['Chunk_Index'] = chunks_indices

        #  Group by chunk index
        grouped_parsed_df = parsed_df.groupby('Chunk_Index')

        for ck_idx, group in grouped_parsed_df:
            if ck_idx == -1:  # Not a chunk
                continue
            chunk_df = group.drop(columns='Chunk_Index').reset_index(drop=True)
            parsed_chunk_data = ParsedData(parsed_data.label, chunk_df,
                                           parsed_data.train_test,
                                           parsed_data.rec_number,
                                           parsed_data.metadata)  # Still, provisional per-recording label. The label itself will be added at the preprocessing stage
            data_with_chunks_list.append(parsed_chunk_data)

    return data_with_chunks_list


def run_volume_tagger(data_with_chunks_list: List[ParsedData]):
    for d in data_with_chunks_list:
        d.data = pd.concat([d.data, VolumeChunkTagger.tag_chunk_by_volume(d.data)], axis=1)
    return data_with_chunks_list


def run_preprocessing(data_with_chunks_list: List[ParsedData], config):
    model_input_list = []

    if config['model_name'] == 'CLT' or config['model_name'] == 'CommandLevelLSTM' or config['model_name'] == 'CommandLevelLSTMContinuous':
        preprocessor = CLTTokenizer.CLTTokenizer()
    elif config['model_name'] == 'PLT' or config['model_name'] == 'UNet':
        preprocessor = PLTTokenizer.PLTTokenizer()
    elif config['model_name'] == 'RF':
        preprocessor = RFFeaturizer.RFFeaturizer()
    else:
        preprocessor = DeftPunkFeaturizer.DeftPunkFeaturizer()

    for chunk_data_element in data_with_chunks_list:
        current_features = preprocessor.generate_per_chunk(chunk_data_element)
        rec_label = chunk_data_element.label

        #  Update chunk label
        if config['model_name'] == 'PLT' or config['model_name'] == 'UNet':
            current_chunk_label = get_chunk_label(chunk_data_element.data, config['model_name'], preprocessor)
        else:
            current_chunk_label = get_chunk_label(chunk_data_element.data, config['model_name'])
        chunk_data_element.label = current_chunk_label

        current_model_input = ModelInput(chunk_data_element, current_features, rec_label)

        model_input_list.append(current_model_input)

    return model_input_list


def get_chunk_label(chunk_df, model, preprocessor=None):
    if model == 'RF' or model == 'DeftPunk':
        if chunk_df.Label.any():
            return 1
        return 0
    elif model == 'PLT' or model == 'UNet':
        return preprocessor.get_plt_labels(chunk_df)
    else:
        return chunk_df.Label


def train_and_infer(data_with_features_list: List[ModelInput], config):
    if config['model_name'] == 'CLT' or config['model_name'] == 'PLT' or config['model_name'] == 'CommandLevelLSTM' or config['model_name'] == 'CommandLevelLSTMContinuous' or config['model_name'] == 'UNet':
        mdl_mgr = TorchManager(data_with_features_list, config)
    else:
        mdl_mgr = TreeManager(data_with_features_list, config)

    mdl_mgr.train()

    results_split = mdl_mgr.inference()

    return results_split


def divide_train_test(dataset: List[ModelInput]):
    train_set = []
    test_set = []

    for element in dataset:
        if element.parsed_data_element.train_test == "Train":
            train_set.append(element)
        else:
            test_set.append(element)

    return train_set, test_set


def obtain_features_and_labels(dataset):
    if type(dataset) != list:
        X = np.array(dataset.features)
        y = np.array(dataset.parsed_data_element.label)
    else:
        X = np.array([element.features for element in dataset])
        y = np.array([element.parsed_data_element.label for element in dataset])

    return X, y


def fracreg_to_label(batch_labels, model_name):
    if model_name == "PLT" or model_name == "UNet":
        lab = batch_labels[:, 0:2, :].sum(axis=1).mean(axis=1) > 0
    else:  # CLT etc.
        lab = (batch_labels.mean(axis=1) if len(
            batch_labels.shape) > 1 else batch_labels.mean()) > 0
    try:
        return lab[0]
    except:
        return lab


def fracreg_to_predict_prob(predict_prob, model_name):
    if model_name == "PLT" or model_name == "UNet":
        pred_prob = predict_prob[:, 0:2, :].sum(axis=1).mean(axis=1)
    else:  # CLT
        pred_prob = predict_prob.mean() if len(predict_prob.shape) > 1 else predict_prob
    try:
        return pred_prob[0]
    except:
        return pred_prob


def extract_train_and_test_for_demo_mode(dataset: List[ParsedData]):
    num_train_chunks_rw = 5
    num_train_chunks_benign = 5
    num_test_chunks_rw = 2
    num_test_chunks_benign = 2

    random.seed(0)

    # Filter objects into groups
    train_rw = [obj for obj in dataset if obj.train_test == "Train" and obj.label == 1]
    train_benign = [obj for obj in dataset if obj.train_test == "Train" and obj.label == 0]
    test_rw = [obj for obj in dataset if obj.train_test == "Test" and obj.label == 1]
    test_benign = [obj for obj in dataset if obj.train_test == "Test" and obj.label == 0]

    # Randomly sample the required number of objects
    selected_train_rw = random.sample(train_rw, min(num_train_chunks_rw, len(train_rw)))
    selected_train_benign = random.sample(train_benign, min(num_train_chunks_benign, len(train_benign)))
    selected_test_rw = random.sample(test_rw, min(num_test_chunks_rw, len(test_rw)))
    selected_test_benign = random.sample(test_benign, min(num_test_chunks_benign, len(test_benign)))

    demo_list = selected_train_rw + selected_train_benign + selected_test_rw + selected_test_benign

    return demo_list


def clopper_pearson(x, n, alpha=0.05):
    """
    Estimate the confidence interval for a sampled Bernoulli random variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    """
    b = stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi


def evaluate_results(results_input: pd.DataFrame, config: Dict):
    # Split dataframe to eval and non-eval using the gss object of sklearn
    # gss.split is an iterator that contains random eval/test splits as per the gss input parameters

    if config['demo_mode']:
        num_splits = 1
        test_size = 0.5
    else:
        num_splits = 5
        test_size = 0.67

    gss = GroupShuffleSplit(n_splits=num_splits, test_size=test_size, random_state=42)
    results = results_input.copy()

    # Loop over all splits
    kpis_splits = []
    split_number = 0
    for i, (val_index, test_index) in enumerate(gss.split(X=None, y=None, groups=results_input['rec_idx'])):
        split_number += 1

        eval_df: pd.DataFrame = results.iloc[val_index, :]  # the eval df on which we calculate the workpoint
        test_df: pd.DataFrame = results.iloc[test_index, :]  # the non-eval df on which we evaluate performance at the workpoint

        if config['demo_mode']:
            thresh = 0.5
        else:
            # Calculate the threshold on the eval set
            thresh: float = thresh_by_far_plus_ci(eval_df, 0.01)

        # Measure key performance indicators (kpis) on the non-eval set
        kpis = evaluate_workpoint(test_df, thresh=thresh)

        kpis_splits.append(kpis)

    df_results = obtain_results(kpis_splits)

    return df_results


def evaluate_workpoint(df: pd.DataFrame, thresh: float, percentile: float = 0.75) -> Dict:
    kpis = {}

    if len(set(df['target'])) < 2:
        df.loc[df.index[0], 'target'] = 1  # For demo mode

    mdr, far, sample_weights = calculate_volume_rates(df, thresh)
    kpis.update({'MDR': mdr})

    #   Calculate the AUC of the
    auc_score: float = calculate_auc(df, sample_weights)
    kpis.update({'AUC': auc_score})

    # Measure the MBD per each of the ransomware recordings that was detected and its q-th percentile
    mbd: np.ndarray = calculate_mbd(df, thresh)
    #   Measure the q-th percentile of MBD
    kpis.update({'MBD_Q': np.quantile(mbd, q=percentile)})

    return kpis


def obtain_tpr_and_fpr(df: pd.DataFrame) -> tuple:
    return roc_curve(df['target'], df['pred_prob'])


def thresh_by_far_plus_ci(df: pd.DataFrame, far_plus_ci_spec: float) -> float:
    fpr, tpr, thresholds = obtain_tpr_and_fpr(df)
    neg_ground_truth = (df['target'] == 0).sum()
    fp = fpr * neg_ground_truth
    fpr_upper = np.array([clopper_pearson(x, neg_ground_truth)[1] for x in fp])
    # Set the threshold to be such that the fpr <= FAR_spec
    return find_thresh_for_best_tpr_within_far_spec(fpr_upper, tpr, thresholds, far_plus_ci_spec)


def find_thresh_for_best_tpr_within_far_spec(fpr: np.array, tpr: np.array, thresholds: np.array, far_spec: float) -> float:
    # Set the threshold to be such that the fpr <= FAR_spec
    ix = np.argmax(tpr[fpr <= far_spec])
    thresh = thresholds[ix]
    if not 0.0 <= thresh <= 1.0:
        print(f"Fpr: {fpr} resulted in invalid threshold: {thresh} . Note that invalid threshold can result in degraded results!")
    return thresh


def calculate_auc(df: pd.DataFrame, sample_weights) -> float:
    auc_score: float = roc_auc_score(y_true=df['target'], y_score=df['pred_prob'], sample_weight=sample_weights)
    if np.isnan(auc_score):
        auc_score = 0
    return auc_score


def calculate_mbd(df: pd.DataFrame, thresh: float = 0.5) -> np.ndarray:
    mbd: np.ndarray = megabyte_to_detect_per_recording(df, thresh)
    return mbd


def megabyte_to_detect_per_recording(df: pd.DataFrame, thresh: float) -> np.array:
    # Filter to calculate volume only on ransomware recordings
    ransomware_recs = df['recording_label'] == 1
    if ransomware_recs.sum() == 0:  # No RW
        return np.array(df['volume'].sum())
    mbd_write = calculate_written_mbd(df[ransomware_recs], thresh)

    return np.array(mbd_write.to_list())


def calculate_written_mbd(df: pd.DataFrame, thresh: float) -> pd.Series:
    df_new = df.copy(deep=True).reset_index()
    # Find the chunks which are correctly predicted as ransomware
    df_new.loc[:, 'correct pred'] = (df_new['pred_prob'] >= thresh) & df_new['target']
    # Filter out recordings that were not detected
    # (only the groups with an aggregated correct pred == True were detected)
    is_recording_detected = df_new.groupby('rec_idx')['correct pred'].transform(lambda x: any(x))
    df_new = df_new[is_recording_detected]
    # Calculate the first chunk correctly detected
    df_new['write_rw_mb'] = df_new['rsw_write_volume'] * NLB_to_MB
    # Step 2: Compute cumulative sum for each 'Recording_Name'
    df_new['cumsum_mb'] = df_new.groupby('rec_idx')['write_rw_mb'].cumsum()
    indx_first_detection = df_new[df_new['correct pred']].groupby('rec_idx').first()['index']
    # Calculate the first chunk of a recording
    indx_first_chunk = df_new.groupby('rec_idx').first()['index']
    df_first_detected_chunk = indx_first_detection - indx_first_chunk + 1
    df_first_detected_chunk.name = 'first correct pred'
    return df_new[df_new['correct pred']].groupby('rec_idx').first()['cumsum_mb']


def calculate_volume_rates(df: pd.DataFrame, thresh: float = 0.5) -> Tuple[float, float, np.ndarray]:
    sample_weight = calculate_sample_weights(df)
    mdr, far, precision, recall, f1 = calculate_classification_metrics_chunk_level(df, thresh=thresh, sample_weight=sample_weight)
    if np.isnan(mdr):
        mdr = 1
    if np.isnan(far):
        mdr = 1
    return mdr, far, sample_weight


def calculate_sample_weights(df: pd.DataFrame) -> np.array:
    #   Caclulate the sample weights being the following:
    #       weight = V_chunk for chunks that are pure benign
    #       weight = V_chunk_malicious for chunks that are ransomware
    df = df.fillna(0)
    weight = (1 - df['target']) * df['volume'] + \
             df['target'] * (df['rsw_read_volume'] + df['rsw_write_volume'])
    return weight


def calculate_classification_metrics_chunk_level(df: pd.DataFrame, thresh: float = 0.5, sample_weight: np.array = None) -> Tuple[float, float, float, float, float]:
    tn, fp, fn, tp = confusion_matrix(
        y_true=df['target'], y_pred=df['pred_prob'] >= thresh,
        sample_weight=sample_weight
    ).ravel()

    mdr = fn / (tp + fn)
    far = fp / (tn + fp)
    precision = tp / (fp + tp)
    recall = 1.0 - mdr
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return mdr, far, precision, recall, f1


def obtain_results(results_split) -> pd.DataFrame:
    """ Receives results per split and returns results (mean and std) as a DataFrame """
    mdr = []
    auc = []
    mbd_q = []

    for res in results_split:
        mdr.append(res['MDR'])
        auc.append(res['AUC'])
        mbd_q.append(res['MBD_Q'])

    mdr = np.array(mdr)
    auc = np.array(auc)
    mbd_q = np.array(mbd_q)

    df_results = pd.DataFrame({"MDR Mean": [mdr.mean()], "AUC Mean": [auc.mean()], "MBD_Q Mean": [mbd_q.mean()],
                               "MDR STD": [mdr.std()], "AUC STD": [auc.std()], "MBD_Q STD": [mbd_q.std()]})

    return df_results

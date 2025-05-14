import numpy as np
import pandas as pd


def is_ransomware_process(rw_processes_name_and_pids, x):
    return x in rw_processes_name_and_pids


def process_name_id(x):
    w = []
    for y in x.split('/')[1:]:
        z = y.split(' ')
        w.append(z[0] + z[1][z[1].find('(') + 1:z[1].find(')')])
    return w


def black_files(df):
    return list(set(df[df['RW_process']]['Path'].to_list()))


def black_slba_in_file(df, filen):
    splitter = '/' if '/' in df['Path'].to_list()[0] else '\\'
    black_slba_list = set(df[df['RW_process'] &
                             (df['Path'].apply(lambda x: x.split(splitter)[-1] == filen))]['slba'].to_list())
    return df['slba'].isin(list(black_slba_list))


def mark_potentially_blacked_rows(df, name):
    if name == 'black_file':
        special_files = {'unknown (0x0)', '$mft', '$logfile', '$bitmap'}
        black_list = black_files(df)
        splitter = '/' if (len(black_list) and '/' in black_list[0]) else '\\'
        file_list = [x for x in black_list if x.split(splitter)[-1] not in special_files]
        df[name] = df['Path'].isin(file_list)
    elif name == 'black_slba_mft':
        df[name] = black_slba_in_file(df, '$mft')
    elif name == 'black_slba_logfile':
        df[name] = black_slba_in_file(df, '$logfile')
    elif name == 'black_slba_bitmap':
        df[name] = black_slba_in_file(df, '$bitmap')

    return df


def first_relevant_benign_timestamp(x):
    if ~np.isnan(x.t_benign).any():
        t = x.t_benign[x.t_rw_max < x.t_benign]
        if len(t) > 0:
            return t.min()
        else:
            return np.inf
    else:
        return np.inf


def mark_blacked_rows_wo_apply(df, name):
    if '_slba_' in name:
        colname = 'RW_or_RWsys_access_to_' + name
    else:
        colname = 'RWsys_access_to_' + name

    by = 'slba' if '_slba_' in name else 'Path'

    ser_t_rw = df[df[name] & df['RW_process']].groupby(by=by).agg(t_rw_min=('timestamp', 'min'),
                                                                  t_rw_max=('timestamp', 'max'))
    ser_t_benign = df[df[name] & df['benign']].groupby(by=by)['timestamp'].apply(np.hstack)
    ser_t_benign.rename('t_benign', inplace=True)
    df_t = pd.merge(pd.DataFrame(ser_t_rw), pd.DataFrame(ser_t_benign), on=by, how='outer')
    df_t.loc[:, 't_benign_first'] = df_t.apply(lambda x: first_relevant_benign_timestamp(x), axis=1)

    df_tmp = df[[by, 'timestamp', 'Process Name', 'RW_process']].join(df_t, on=by)

    if '_slba_' in name:
        df.loc[:, colname] = (df_tmp.timestamp >= df_tmp.t_rw_min) & (df_tmp.timestamp < df_tmp.t_benign_first) & \
                             ((df_tmp['Process Name'] == 'System') | df_tmp['RW_process'])
    else:
        df.loc[:, colname] = (df_tmp.timestamp >= df_tmp.t_rw_min) & (df_tmp.timestamp < df_tmp.t_benign_first) & \
                             (df_tmp['Process Name'] == 'System')
    return df


def create_benign_tag_columns(raw_df) -> None:
    raw_df['tagged_as_rsw'] = np.uint8(0)
    raw_df['tagged_as_rsw_type'] = np.uint8(0)
    return


def create_tagged_rsw_col_causal(raw_df, launch_time, rw_processes, benign_processes) -> None:
    raw_df.loc[:, 'RW_process'] = raw_df['Process name (ID)']. \
        apply(lambda x: is_ransomware_process(rw_processes, x))
    raw_df.loc[:, 'benign'] = raw_df['Process Name'].isin(benign_processes)

    raw_df['RWsys_access_to_black_file'] = False
    raw_df['RW_or_RWsys_access_to_black_slba_mft'] = False
    raw_df['RW_or_RWsys_access_to_black_slba_logfile'] = False
    raw_df['RW_or_RWsys_access_to_black_slba_bitmap'] = False
    for name in ['black_file', 'black_slba_logfile', 'black_slba_mft', 'black_slba_bitmap']:
        raw_df = mark_potentially_blacked_rows(raw_df, name)
        raw_df = mark_blacked_rows_wo_apply(raw_df, name)

    raw_df['tagged_as_rsw'] = np.zeros(raw_df.shape[0])
    raw_df['tagged_as_rsw_type'] = np.zeros(raw_df.shape[0])
    raw_df.loc[raw_df['RW_process'], 'tagged_as_rsw_type'] = 1.0
    raw_df.loc[raw_df['RWsys_access_to_black_file']
               & (raw_df['timestamp'] >= launch_time), 'tagged_as_rsw_type'] = 2.0
    raw_df.loc[raw_df['RW_or_RWsys_access_to_black_slba_mft']
               & (raw_df['timestamp'] >= launch_time), 'tagged_as_rsw_type'] = 3.0
    raw_df.loc[raw_df['RW_or_RWsys_access_to_black_slba_logfile']
               & (raw_df['timestamp'] >= launch_time), 'tagged_as_rsw_type'] = 4.0
    raw_df.loc[raw_df['RW_or_RWsys_access_to_black_slba_bitmap']
               & (raw_df['timestamp'] >= launch_time), 'tagged_as_rsw_type'] = 5.0

    raw_df['RW_label'] = (raw_df['RW_process'] |
                          raw_df['RWsys_access_to_black_file'] |
                          raw_df['RW_or_RWsys_access_to_black_slba_mft'] |
                          raw_df['RW_or_RWsys_access_to_black_slba_logfile'] |
                          raw_df['RW_or_RWsys_access_to_black_slba_bitmap']) & \
                         (raw_df['timestamp'] >= launch_time)

    raw_df.loc[raw_df['RW_label'], 'tagged_as_rsw'] = 1.0

    return


def rw_processes_from_tree(df_tree):
    rw_processes_name_and_pids = set(
        df_tree.loc[df_tree['Process Tree'].str.contains('sample_'), 'Process Tree'].apply(process_name_id).sum()
    )
    return list(rw_processes_name_and_pids)


benign_processes = ['python.exe', '7z.exe', 'powershell.exe', 'TiWorker.exe',
                    'git.exe', 'git-remote-https.exe',
                    'pycharm-community-2022.1.4.exe',
                    'sdelete64.exe', 'fsutil.exe']


def run_labeling(df_recording: pd.DataFrame, df_process_tree: pd.DataFrame, rec_label: int) -> pd.DataFrame:
    """
    :param df_recording: The recording DataFrame
    :param df_process_tree: The process tree DataFrame
    :param rec_label: The recording label (0 for benign, 1 from RW). Can be obtained from a recording's metadata
    :return: A DataFrame with a "Label" column (or an empty DataFrame, if failed)
    """

    df_recording = df_recording.rename(columns={"Offset": 'slba', "Size": 'nlb', "OpCode": 'opcode', 'Timestamp': 'timestamp'})
    df_recording["slba"] = df_recording["slba"].apply(lambda val: int(val // 512))
    df_recording["nlb"] = df_recording["nlb"].apply(lambda val: int(val // 512))

    if rec_label == 0:
        create_benign_tag_columns(df_recording)
    elif rec_label == 1:
        # Validate inputs
        if df_process_tree is None:
            print(f"df_process_tree invalid: {df_process_tree is None}")
            return pd.DataFrame()

        rw_processes = rw_processes_from_tree(df_process_tree)
        t0 = -np.inf
        print(f"rw_processes: {rw_processes}")
        create_tagged_rsw_col_causal(df_recording, t0, rw_processes, benign_processes)
    else:
        print(f"Invalid label given: {rec_label}")

    df_recording['tagged_as_rsw'] = df_recording['tagged_as_rsw'].astype(np.uint8)
    df_recording = df_recording.rename(columns={'tagged_as_rsw': 'Label'})

    return df_recording['Label']

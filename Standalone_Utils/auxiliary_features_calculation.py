import pandas as pd
import numpy as np
from AuxPreProcessing import ReadAddressesSegmentsChunkedDict

aux_related_params = {'frequency_of_hash_probe': 1000e10, 'nlb_range': 4096, 'hash_size': 55555, 'balance_by': 'segments', 'max_num_of_segments': 5e4, 'max_num_of_lbas': 5e6}


def calculate_auxiliary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the auxiliary features.
    :param df: DataFrame of NVMe commands (columns: Timestamp, Offset, Size, OpCode)
    :returns: DataFrame with the auxiliary features
    """
    df = df.rename(columns={"Offset": 'slba', "Size": 'nlb', "OpCode": 'opcode', 'Timestamp': 'timestamp'})
    df["slba"] = df["slba"].apply(lambda val: int(val // 512))
    df["nlb"] = df["nlb"].apply(lambda val: int(val // 512))

    war_lengths, rar_lengths, raw_lengths, waw_lengths = [], [], [], []
    war_lapses, rar_lapses, raw_lapses, waw_lapses = [], [], [], []
    war_table_lengths, raw_table_lengths = [0], [0]

    read_address_manager = ReadAddressesSegmentsChunkedDict(aux_related_params)
    write_address_manager = ReadAddressesSegmentsChunkedDict(aux_related_params)

    bins_0 = np.linspace(0, 12, 13)
    bins_minus_inf = np.array([-np.inf] + list(bins_0)[:-1])

    num_segments_hist, nlb_segments_mean_hist, nlb_segments_std_hist = [], [], []
    num_holes_hist, nlb_holes_mean_hist, nlb_holes_std_hist = [], [], []
    index_of_wardb_probe = []
    df_unpclipped_nlb = df['nlb'].copy()
    df['nlb'] = np.clip(df['nlb'], 1, 4986)

    for i in range(len(df)):
        war_length = 0
        rar_length = 0
        raw_length = 0
        waw_length = 0

        war_lapse = 0
        rar_lapse = 0
        raw_lapse = 0
        waw_lapse = 0

        #  ################################# WRITE ####################################
        if df['opcode'].iloc[i] == 1:
            war_length, war_lapse = read_address_manager.lbas_write(df['slba'].iloc[i],
                                                                    df['nlb'].iloc[i],
                                                                    df['timestamp'].iloc[i])

            waw_length, waw_lapse = write_address_manager.lbas_read(df['slba'].iloc[i],
                                                                    df['nlb'].iloc[i],
                                                                    df['timestamp'].iloc[i])

        #  ################################# READ ####################################
        if df['opcode'].iloc[i] == 2:
            rar_length, rar_lapse = read_address_manager.lbas_read(df['slba'].iloc[i],
                                                                   df['nlb'].iloc[i],
                                                                   df['timestamp'].iloc[i])

            raw_length, raw_lapse = write_address_manager.lbas_write(df['slba'].iloc[i],
                                                                     df['nlb'].iloc[i],
                                                                     df['timestamp'].iloc[i])

        if (i + 1) % aux_related_params['frequency_of_hash_probe'] == 0:
            war_data = read_address_manager.probe_war_zone()
            (num_segments, num_holes, nlb_segment_mean, nlb_segment_std, nlb_hole_mean, nlb_hole_std) = war_data
            num_segments_hist.append(np.histogram(np.log2(num_segments), bins=bins_0, density=True)[0])
            nlb_segments_mean_hist.append(np.histogram(np.log2(nlb_segment_mean), bins=bins_0, density=True)[0])
            nlb_segments_std_hist.append(np.histogram(np.log2(nlb_segment_std), bins=bins_minus_inf, density=True)[0])
            num_holes_hist.append(np.histogram(np.log2(num_holes), bins=bins_minus_inf, density=True)[0])
            nlb_holes_mean_hist.append(np.histogram(np.log2(nlb_hole_mean), bins=bins_0, density=True)[0])
            nlb_holes_std_hist.append(np.histogram(np.log2(nlb_hole_std), bins=bins_minus_inf, density=True)[0])
            index_of_wardb_probe.append(i)

        read_address_manager.balance_size()
        write_address_manager.balance_size()

        war_lengths.append(war_length)
        rar_lengths.append(rar_length)
        waw_lengths.append(waw_length)
        raw_lengths.append(raw_length)

        war_lapses.append(war_lapse)
        rar_lapses.append(rar_lapse)
        waw_lapses.append(waw_lapse)
        raw_lapses.append(raw_lapse)

        war_table_lengths.append(read_address_manager.get_size())
        raw_table_lengths.append(write_address_manager.get_size())

    df['WAR'] = war_lengths
    df['RAR'] = rar_lengths
    df['RAW'] = raw_lengths
    df['WAW'] = waw_lengths

    df['WAR Lapse'] = war_lapses
    df['RAR Lapse'] = rar_lapses
    df['RAW Lapse'] = raw_lapses
    df['WAW Lapse'] = waw_lapses

    df.fillna(0, inplace=True)
    df['nlb'] = df_unpclipped_nlb.copy()

    return df[['WAR', 'RAR', 'RAW', 'WAW', 'WAR Lapse', 'RAR Lapse', 'RAW Lapse', 'WAW Lapse']]

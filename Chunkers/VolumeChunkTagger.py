import numpy as np
import pandas as pd
from Misc.Constants import *


# This tagger creates a tag that counts the percentage of the NLB volume of the ransomware commands,
# separately for read and for write commands
# Output is a np array with two entries:
#     - the percentage of NLB associated with ransomware read commands out of the overall read volume
#     - the percentage of NLB associated with ransomware write commands out of the overall write volume
#     Note: the percentage = 0.0 for pure benign and percentage = 1.0 for pure ransomware


def tag_chunk_by_volume(data_chunk: pd.DataFrame) -> np.array:
    rw_NLB_read, rw_NLB_write = np.nan, np.nan

    condition_read = data_chunk['OpCode'] == NVMeOpCodeREAD
    NLB_read = data_chunk[condition_read]['NLB'].to_numpy().sum()
    NLB_write = data_chunk[~condition_read]['NLB'].to_numpy().sum()
    chunk_volume = NLB_read + NLB_write

    if 'Label' in data_chunk.columns:
        condition_rw = data_chunk['Label'] == 1.0
        rw_NLB_read = data_chunk[condition_read & condition_rw]['NLB'].to_numpy().sum()
        rw_NLB_write = data_chunk[~condition_read & condition_rw]['NLB'].to_numpy().sum()

    return pd.DataFrame({"volume": [chunk_volume], 'rsw_read_volume': [rw_NLB_read / chunk_volume], 'rsw_write_volume': [rw_NLB_write / chunk_volume]}, index=data_chunk.index)

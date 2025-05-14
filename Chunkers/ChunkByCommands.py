from typing import List
import pandas as pd
from Misc.ParsedData import ParsedData


class ChunkByCommands:
    def __init__(self, parsed_data: List[ParsedData], config):
        self.parsed_data = parsed_data
        self.commands_per_slice = config['CLT']['num_tokens'] // 2

    def create_chunk_indices_series(self, data: pd.DataFrame) -> pd.Series:
        commands_per_chunk = self.commands_per_slice
        index_list = data.index.tolist()
        df_size = len(index_list)
        chunk_series = pd.Series([-1] * df_size, dtype=int)

        # Iterate over selected indices in chunks of 'chunk_size'
        cluster_id = 0
        for i in range(0, len(index_list), commands_per_chunk):
            indices = index_list[i:i + commands_per_chunk]

            # Check if the chunk has exactly chunk_size elements
            if len(indices) == commands_per_chunk:
                # Assign the current cluster_id to each index in the chunk
                chunk_series[indices] = cluster_id
                cluster_id += 1

        return chunk_series

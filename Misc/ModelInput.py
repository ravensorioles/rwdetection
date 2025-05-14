import numpy as np
from Misc.ParsedData import ParsedData


class ModelInput:

    def __init__(self, parsed_data_element: ParsedData, features: np.ndarray, rec_label: int):
        self.parsed_data_element = parsed_data_element
        self.features = features
        self.rec_label = rec_label

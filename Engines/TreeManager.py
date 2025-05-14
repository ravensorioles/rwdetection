import pandas as pd
from typing import List, Dict
from Misc.ModelInput import ModelInput
from Models.RFModel import RFModel
from Models.DeftPunkModel import DeftPunkModel
from Misc import Utils


class TreeManager:

    def __init__(self,
                 dataset: List[ModelInput],
                 config: Dict
                 ):

        if config['model_name'] == 'RF':
            self.model = RFModel(config)
        else:
            self.model = DeftPunkModel(config)

        self.dataset_train, self.dataset_test = Utils.divide_train_test(dataset)
        self.X_train, self.y_train = Utils.obtain_features_and_labels(self.dataset_train)
        self.X_test, self.y_test = Utils.obtain_features_and_labels(self.dataset_test)
        self.demo_mode = config['demo_mode']

        if self.demo_mode:
            self.num_splits = 1
            self.test_size = 0.5
        else:
            self.num_splits = 5
            self.test_size = 0.67

    def train(self):
        self.ML_model = self.model.get_pipeline()
        self.ML_model.fit(self.X_train, self.y_train)

    def inference(self):
        rec_idx = [x.parsed_data_element.rec_number for x in self.dataset_test]
        rec_label = [x.rec_label for x in self.dataset_test]
        volume = [x.parsed_data_element.data['volume'].sum() for x in self.dataset_test]
        rsw_read_volume = [x.parsed_data_element.data['rsw_read_volume'].sum() for x in self.dataset_test]
        rsw_write_volume = [x.parsed_data_element.data['rsw_write_volume'].sum() for x in self.dataset_test]

        results = pd.DataFrame({'pred_prob': self.ML_model.predict(self.X_test)})
        results['target'] = self.y_test
        results['rec_idx'] = rec_idx
        results['recording_label'] = rec_label
        results['volume'] = volume
        results['rsw_read_volume'] = rsw_read_volume
        results['rsw_write_volume'] = rsw_write_volume

        return results

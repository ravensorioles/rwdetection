import numpy as np
import torch
from torch import optim
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from Models.CLTModel import CLTModel
from Models.PLTModel import PLTModel
from Models.CommandLevelLSTMModel import CommandLevelLSTMModel
from Models.CommandLevelLSTMContinuousModel import CommandLevelLSTMContinuous
from Models.UNetModel import UNetModel
from Misc.ModelInput import ModelInput
from Misc.TorchDataset import TorchDataset
from Misc import Utils
from torch.utils.data import DataLoader


def transformer_collate_fn(batch):
    return batch


class TorchManager:

    def __init__(self, dataset: List[ModelInput], config: Dict,
                 learning_rate: float = 0.0001,
                 num_epochs=200,
                 ):
        self.config = config
        self.demo_mode = config['demo_mode']
        self.dataset = dataset
        self.dataset_train_, self.dataset_test_ = Utils.divide_train_test(dataset)
        self.dataset_train = TorchDataset(self.dataset_train_)
        self.dataset_test = TorchDataset(self.dataset_test_)

        # set model & schedule the optimizer
        self.model_name = config['model_name']
        if self.model_name == 'CLT':
            scheduler = {'step': 30, 'gamma': 0.9}
            self.model = CLTModel(config)
            self.batch_size = 64
            self.reduce_factor = int((250 / (config['CLT']['num_tokens'] // 2)) * 66)
        elif self.model_name == 'PLT':
            scheduler = {'step': 5000, 'gamma': 0.8}
            self.model = PLTModel(config)
            self.batch_size = 256
        elif self.model_name == 'CommandLevelLSTM':
            scheduler = {'step': 30, 'gamma': 0.8}
            self.model = CommandLevelLSTMModel(config)
            self.batch_size = 64
            self.reduce_factor = int((250 / (config['CommandLevelLSTM']['num_tokens'] // 2)) * 66)
        elif self.model_name == 'UNet':
            scheduler = {'step': 5000, 'gamma': 0.8}
            self.model = UNetModel(config)
            self.batch_size = 256
        else:
            scheduler = {'step': 30, 'gamma': 0.8}
            self.model = CommandLevelLSTMContinuous(config)
            self.batch_size = 64
            self.reduce_factor = int((250 / (config['CommandLevelLSTMContinuous']['num_tokens'] // 2)) * 66)
        self.num_epochs = num_epochs
        self.test_size = 0.67  # For splits
        self.num_splits = 5
        if self.demo_mode:
            self.batch_size = 2
            self.num_epochs = 1
            self.num_splits = 1
            self.test_size = 0.5

        self.dataset_train_loader = DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, collate_fn=transformer_collate_fn)
        self.dataset_test_loader = DataLoader(dataset=self.dataset_test, batch_size=self.batch_size, collate_fn=transformer_collate_fn)
        self.num_train_batches = self.dataset_train_.__len__() // self.batch_size
        self.num_test_batches = self.dataset_test_.__len__() // self.batch_size
        self.epoch_num = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = learning_rate
        self.loss = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler['step'],
                                                   gamma=scheduler['gamma'])
        self.reduce_factor = 1 if config['model_name'] == "PLT" or config['model_name'] == "UNet" else self.reduce_factor  # factor 1 -> no reduction

    def train(self):
        # Loop for each Epoch
        while self.epoch_num < self.num_epochs:
            # Perform a training iteration
            self._training_iteration_per_db()

            self.scheduler.step()
            self.epoch_num += 1

    def inference(self):
        chunk_results = self._evaluation_iteration()

        return chunk_results

    def _training_iteration_per_batch(self, batch: ModelInput) -> torch.Tensor:
        """
        """
        states_, labels_ = Utils.obtain_features_and_labels(batch)
        states = torch.Tensor(states_).to(self.device)
        labels = torch.Tensor(labels_).to(self.device)

        res = self.model(states).squeeze()
        if self.model_name == 'PLT' or self.model_name == 'UNet':
            labels = labels[:, :2, :]
        loss = self.loss(res, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _training_iteration_per_db(self):
        """
        """
        # Create DatasetLogic Iteration
        ds_iter = iter(self.dataset_train_loader)

        # Keep track of our losses
        losses = []

        # Loop per batch in our dataset
        for _ in tqdm(range(self.num_train_batches)):
            curr_batch = next(ds_iter)

            loss = self._training_iteration_per_batch(curr_batch)
            losses.append(loss.cpu().detach().numpy())

    def _evaluation_iteration(self):

        y_preds, y_targs = [], []

        if self.config['model_name'] == 'CommandLevelLSTMContinuous':
            with torch.no_grad():
                nchunks = 100000 // self.config['CommandLevelLSTMContinuous']['num_tokens']
                for curr_rec in tqdm(self.dataset_test):
                    states_, labels_ = Utils.obtain_features_and_labels(curr_rec)
                    states = torch.Tensor(states_).to(self.device)

                    if labels_.ndim <= 2:
                        labels_ = labels_[np.newaxis, :]
                    if states.ndimension() <= 2:
                        states = states.unsqueeze(0)

                    for j in range(int(np.ceil(len(states) / nchunks))):

                        chunkified_states = states[j * nchunks: min((j + 1) * nchunks, len(states))].ravel()[np.newaxis, :]
                        chunkified_labels = labels_[j * nchunks: min((j + 1) * nchunks, len(states))]

                        res = self.model((chunkified_states, bool(j))).detach().cpu().numpy().reshape((-1, self.config['CommandLevelLSTMContinuous']['num_tokens'] // 2))

                        y_targs.append(Utils.fracreg_to_label(chunkified_labels, self.model_name).tolist())
                        y_preds.append(Utils.fracreg_to_predict_prob(res, self.model_name).tolist())

                rec_idx = [x.parsed_data_element.rec_number for x in self.dataset_test]  # TODO ?
                volume = [x.parsed_data_element.data['volume'].sum() for x in self.dataset_test]
                rsw_read_volume = [x.parsed_data_element.data['rsw_read_volume'].sum() for x in self.dataset_test]
                rsw_write_volume = [x.parsed_data_element.data['rsw_write_volume'].sum() for x in self.dataset_test]
                rec_label = [x.rec_label for x in self.dataset_test]

        else:
            with torch.no_grad():
                for curr_batch in tqdm(self.dataset_test, total=self.num_test_batches):
                    states_, labels_ = Utils.obtain_features_and_labels(curr_batch)
                    states = torch.Tensor(states_).to(self.device)
                    if labels_.ndim <= 2:
                        labels_ = labels_[np.newaxis, :]
                    if states.ndimension() <= 2:
                        states = states.unsqueeze(0)
                    if self.model_name == 'PLT' or self.model_name == 'UNet':
                        labels_ = labels_[:, :2, :]

                    res = self.model(states).detach().cpu().numpy()

                    y_targs.append(Utils.fracreg_to_label(labels_, self.model_name).tolist())
                    y_preds.append(Utils.fracreg_to_predict_prob(res, self.model_name).tolist())

                rec_idx = [x.parsed_data_element.rec_number for x in self.dataset_test]
                volume = [x.parsed_data_element.data['volume'].sum() for x in self.dataset_test]
                rsw_read_volume = [x.parsed_data_element.data['rsw_read_volume'].sum() for x in self.dataset_test]
                rsw_write_volume = [x.parsed_data_element.data['rsw_write_volume'].sum() for x in self.dataset_test]
                rec_label = [x.rec_label for x in self.dataset_test]

        y_targs = [int(item) if isinstance(item, bool) else item for item in y_targs]
        results = pd.DataFrame({'target': y_targs, 'pred_prob': y_preds, 'rec_idx': rec_idx, 'recording_label': rec_label,
                                'volume': volume, 'rsw_read_volume': rsw_read_volume, 'rsw_write_volume': rsw_write_volume})

        return results

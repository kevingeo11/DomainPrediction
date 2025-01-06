import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy import stats

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from peft import LoraConfig, get_peft_model


class RFSurrogate():
    def __init__(self) -> None:

        self.model = RandomForestRegressor(n_estimators=100, criterion='friedman_mse', max_depth=None, min_samples_split=2,
                                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
                                            max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                            n_jobs=None, random_state=1, verbose=1, warm_start=False, ccp_alpha=0.0,
                                            max_samples=None)
    
    def trainmodel(self, X, y, val=None, debug=True):
        '''
            X - embeddings from PLMs (n, features)
            y - shape (n, )
        '''
        _ = self.model.fit(X, y)
        if debug:
            self.print_eval(X, y, label='train')
            if val is not None:
                X_val, y_val = val
                self.print_eval(X_val, y_val, label='val')

    
    def print_eval(self, X, y, label='set'):
        ypred = self.model.predict(X)
        mse = mean_squared_error(ypred, y)
        corr = stats.spearmanr(ypred, y)

        print(f'{label}: mse = {mse}, spearman correlation = {corr.statistic}')

    def predict(self, X):

        return self.model.predict(X)


class RidgeSurrogate():
    def __init__(self, alpha=1.0) -> None:

        self.model = Ridge(alpha=alpha, fit_intercept=True, random_state=1)
    
    def trainmodel(self, X, y, val=None, debug=True):
        '''
            X - embeddings from esm2
            X - shape (n, features)
            y - shape (n, )
        '''
        _ = self.model.fit(X, y)
        if debug:
            self.print_eval(X, y, label='train')
            if val is not None:
                X_val, y_val = val
                self.print_eval(X_val, y_val, label='val')
    
    def print_eval(self, X, y, label='set'):
        ypred = self.model.predict(X)
        mse = mean_squared_error(ypred, y)
        corr = stats.spearmanr(ypred, y)

        print(f'{label}: mse = {mse}, spearman correlation = {corr.statistic}')

    def predict(self, X):

        return self.model.predict(X)
    

class EmbedFunDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPSurrogate(pl.LightningModule):
    def __init__(self, config={'layers': [1280, 2048, 1280, 1], 
                               'epoch': 10, 
                               'batch_size': 16,
                               'patience': 10,
                               'lr': 1e-3,
                               'early_stopping': True,
                               'debug': True}
                ) -> None:
        super().__init__()
        self.config = config

        layers = []
        for i in range(1, len(config['layers'])-1):
            layers.append(nn.Linear(config['layers'][i-1], config['layers'][i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['layers'][-2], config['layers'][-1]))
        self.mlp = nn.Sequential(*layers)

        self.accumulate_batch_loss_train = []
        self.accumulate_batch_loss_val = []
        self.debug = config['debug']

    def forward(self, x):
        x = self.mlp(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat.flatten(), y)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.accumulate_batch_loss_train.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat.flatten(), y)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.accumulate_batch_loss_val.append(loss.item())
    

    def trainmodel(self, X, y, val=None):
        '''
            X - embeddings from esm2
            X - shape (n, features)
            y - shape (n, )
        '''
        
        train_dataset = EmbedFunDataset(X, y)

        val_loader = None
        if val is not None:
            X_val, y_val = val
            val_dataset = EmbedFunDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        callbacks = None
        if self.config['early_stopping']:
            callbacks = []
            earlystopping_callback = EarlyStopping(monitor="val/loss", patience=self.config['patience'], verbose=False, mode="min")
            callbacks.append(earlystopping_callback)

        trainer = pl.Trainer(max_epochs=self.config['epoch'], callbacks=callbacks,
                                accelerator="auto",
                                enable_progress_bar=False,
                                enable_model_summary=True
                                )

        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def on_train_epoch_start(self):
        self.accumulate_batch_loss_train.clear()
        self.accumulate_batch_loss_val.clear()
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.config['print_every_n_epoch'] == 0 and self.debug:
            print(f'Epoch: {self.current_epoch}: train mse: {np.mean(self.accumulate_batch_loss_train)} val mse: {np.mean(self.accumulate_batch_loss_val)}')

    def on_train_end(self):
        print(f'Epoch: {self.current_epoch}: train mse: {np.mean(self.accumulate_batch_loss_train)} val mse: {np.mean(self.accumulate_batch_loss_val)}')

    def predict(self, X):
        '''
            X is numpy array
        '''
        with torch.no_grad():
            y = self(torch.tensor(X))
        return y.numpy().flatten()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])
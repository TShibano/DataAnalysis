#!/usr/bin/env python
# DeepLearning.py: study DeepLearning

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import EarlyStopping
import optuna
from sklearn.datasets import load_breast_cancer
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = 'chrome'
# when png, error occur.


class NeuralNet(pl.LightningModule):

    def __init__(self, n_layers=1, n_mid=1, lr=0.01):
        super().__init__()
        self.n_layers = n_layers
        self.n_mid = n_mid
        self.lr = lr

        self.layers = nn.Sequential()
        for n in range(self.n_layers):
            if n == 0:
                self.layers.add_module(f'fc{n + 1}', nn.Linear(30, self.n_mid))
            else:
                self.layers.add_module(f'fc{n + 1}', nn.Linear(self.n_mid, self.n_mid))
            self.layers.add_module(f'act{n + 1}', nn.ReLU())
            self.layers.add_module(f'bn{n + 1}', nn.BatchNorm1d(self.n_mid))
        self.layers.add_module(f'fc{n_layers + 1}', nn.Linear(self.n_mid, 2))

    def forward(self, x):
        h = self.layers(x)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)
        loss = F.cross_entropy(y, t)
        acc = accuracy(F.softmax(y), t)
        # add log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)
        loss = F.cross_entropy(y, t)
        acc = accuracy(F.softmax(y), t)
        # add log
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('validation_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)
        loss = F.cross_entropy(y, t)
        acc = accuracy(F.softmax(y), t)
        # add log
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def objective(trial, train_loader, val_loader):
    # objective function

    # hyper parameter
    n_layers = trial.suggest_int('n_layers', 1, 5)
    n_mid = trial.suggest_int('n_mid', 2, 20)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    # maximize accuracy for validation data
    pl.seed_everything(0)
    neural_net = NeuralNet(n_layers=n_layers, n_mid=n_mid, lr=lr)
    trainer = pl.Trainer(max_epochs=10, deterministic=True,
                         callbacks=[EarlyStopping(monitor='validation_acc')])
    trainer.fit(neural_net, train_loader, val_loader)

    return trainer.callback_metrics['validation_acc']


def main():
    breast_cancer = load_breast_cancer()
    x = breast_cancer['data']
    t = breast_cancer['target']

    # convert np.array to tensor
    x = torch.from_numpy(x.astype(np.float32))
    t = torch.from_numpy(t.astype(np.int64))

    # dataset
    dataset = torch.utils.data.TensorDataset(x, t)

    # split dataset
    n_train = int(len(dataset)*0.6)
    n_val = int(len(dataset)*0.2)
    n_test = len(dataset) - n_train - n_val
    print([n_train, n_val, n_test])
    pl.seed_everything(0)
    train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    # DataLoader
    pl.seed_everything(0)
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size)

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=30)
    print(study.best_params)
    print('\n')
    print(study.best_value)
    fig = go.Figure(optuna.visualization.plot_optimization_history(study=study))
    fig2 = go.Figure(optuna.visualization.plot_slice(study=study))
    fig.show()
    fig2.show()


if __name__ == '__main__':
    main()

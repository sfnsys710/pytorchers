import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorchers.base import BaseNNRegressor

class NNRegressorEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self, 
        model, 
        loss="mse", 
        optimizer="adam", 
        lr=0.001, 
        epochs=100, 
        batch_size=64, 
        shuffle=True, 
        verbose=True, 
        validation_split=0.1,
        random_state=None
    ):
        """
        PyTorch neural network regressor with sklearn compatibility.
        
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model instance
        loss : str, default="mse"
            Loss function: "mse", "mae", "huber"
        optimizer : str, default="adam"
            Optimizer: "adam", "sgd", "rmsprop"
        lr : float, default=0.001
            Learning rate
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=64
            Batch size
        shuffle : bool, default=True
            Whether to shuffle training data
        verbose : bool, default=True
            Whether to print training progress
        validation_split : float, default=0.1
            Fraction of training data to use for validation
        random_state : int, default=None
            Random seed for reproducibility
        """
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.validation_split = validation_split
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) Training data
        y : array-like of shape (n_samples,) Target values
            
        Returns
        -------
        self : object Fitted estimator
        """
        # Set random seeds for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # instantiate loss and optimizer
        if self.loss == "mse":
            loss = nn.MSELoss()
        if self.optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split, random_state=self.random_state)

        # tensorify data
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=self.shuffle)

        # train model
        self.model.train()
        train_losses, val_losses = [], []

        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                preds = self.model(batch_X)
                batch_loss = loss(preds, batch_y)
                batch_loss.backward()
                optimizer.step()

            with torch.no_grad():
                train_preds, val_preds = self.model(X_train), self.model(X_val)
                train_loss, val_loss = loss(train_preds, y_train).item(), loss(val_preds, y_val).item()
                train_losses.append(train_loss)
                val_losses.append(val_loss)

            if self.verbose:
                print(f"Epoch: {epoch}/{self.epochs} - Train Loss {train_loss} - Val Loss: {val_loss}")
        
        self.train_losses_ = train_losses
        self.val_losses_ = val_losses
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Make predictions for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)Input data
            
        Returns
        -------
        y : ndarray of shape (n_samples,) Predicted values
        """
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X).detach().numpy().ravel()
        
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        params = {
            'model': self.model,
            'loss': self.loss,
            'optimizer': self.optimizer,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'verbose': self.verbose,
            'validation_split': self.validation_split,
            'random_state': self.random_state
        }
        return params
    
    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
class NNRegressor(NNRegressorEstimator):
    def __init__(
        self, 
        input_size, 
        layers=[32, 32, 8], 
        output_size=1,
        loss="mse", 
        optimizer="adam", 
        lr=0.001, 
        epochs=100, 
        batch_size=32, 
        shuffle=True, 
        verbose=True, 
        validation_split=0.2, 
        random_state=None
    ):
        """
        Initialize the NNRegressor class.

        Parameters
        ----------
        input_size : int Number of input features
        layers : list List of hidden layer sizes
        output_size : int Number of output features
        loss : str Loss function to use
        optimizer : str Optimizer to use
        lr : float Learning rate
        epochs : int Number of epochs to train for
        batch_size : int Batch size
        shuffle : bool Whether to shuffle the data
        verbose : bool Whether to print training progress
        """
        self.input_size = input_size
        self.layers = layers
        self.output_size = output_size
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.validation_split = validation_split
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)Input data
        y : array-like of shape (n_samples,) Target values

        Returns
        -------
        self : object
        """
        self.model = BaseNNRegressor(self.input_size, self.layers, self.output_size)
        return super().fit(X, y)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return {
            'input_size': self.input_size,
            'layers': self.layers,
            'output_size': self.output_size,
            'loss': self.loss,
            'optimizer': self.optimizer,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'verbose': self.verbose,
            'validation_split': self.validation_split,
            'random_state': self.random_state
        }

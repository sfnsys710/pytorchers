import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class BaseNNClassifier(nn.Module):
    def __init__(self, input_size, layers=None, n_classes=2):
        nn.Module.__init__(self)
        self.input_size = input_size
        if layers is None:
            layers = [32, 32, 8]
        else:
            self.layers = layers
        self.n_classes = n_classes
        for i in range(len(layers)):
            in_features = input_size if i == 0 else layers[i - 1]
            out_features = layers[i]
            setattr(self, f"fc{i}", nn.Linear(in_features, out_features))
        self.final_linear = nn.Linear(layers[-1], n_classes)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = getattr(self, f"fc{i}")(x)
            x = nn.ReLU()(x)
        x = self.final_linear(x)
        return x


class NNClassifierEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model,
        loss="crossentropy",
        optimizer="adam",
        lr=0.001,
        epochs=100,
        batch_size=64,
        shuffle=True,
        verbose=True,
        validation_split=0.1,
        random_state=None,
        class_weight=None,
    ):
        """
        PyTorch neural network classifier with sklearn compatibility.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model instance
        loss : str, default="crossentropy"
            Loss function: "crossentropy"
        optimizer : str, default="adam"
            Optimizer: "adam"
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
        class_weight : str or dict, default=None
            'balanced' to auto-weight classes, or dict mapping classes to weights
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
        self.class_weight = class_weight

    def fit(self, X, y):
        """
        Fit the model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target class labels

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Set random seeds for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Store classes for sklearn compatibility (REQUIRED)
        self.classes_ = np.unique(y)

        # Map labels to 0, 1 for CrossEntropyLoss
        self.label_encoder_ = LabelEncoder()
        y_array = y.values if hasattr(y, "values") else y
        y_encoded = self.label_encoder_.fit_transform(y_array)

        # Split data into train and validation sets (stratified for imbalanced data)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_encoded,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=y_encoded,
        )

        # Tensorify data - FIX for numpy compatibility
        X_train_array = X_train.values if hasattr(X_train, "values") else X_train
        X_val_array = X_val.values if hasattr(X_val, "values") else X_val
        y_train_array = y_train.values if hasattr(y_train, "values") else y_train
        y_val_array = y_val.values if hasattr(y_val, "values") else y_val

        X_train = torch.tensor(X_train_array, dtype=torch.float32)
        y_train = torch.tensor(y_train_array, dtype=torch.long)  # long for class indices
        X_val = torch.tensor(X_val_array, dtype=torch.float32)
        y_val = torch.tensor(y_val_array, dtype=torch.long)

        dataloader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=self.shuffle
        )

        # Instantiate loss (with class weights if specified)
        if self.loss == "crossentropy":
            if self.class_weight == "balanced":
                # Compute balanced weights
                class_counts = np.bincount(y_train.numpy())
                weights = len(y_train) / (len(class_counts) * class_counts)
                weight_tensor = torch.FloatTensor(weights)
                loss = nn.CrossEntropyLoss(weight=weight_tensor)
            elif isinstance(self.class_weight, dict):
                # Custom weights - map to tensor
                weights = torch.FloatTensor(
                    [self.class_weight.get(cls, 1.0) for cls in range(len(self.classes_))]
                )
                loss = nn.CrossEntropyLoss(weight=weights)
            else:
                loss = nn.CrossEntropyLoss()

        # Instantiate optimizer
        if self.optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Train model
        self.model.train()
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                preds = self.model(batch_X)
                batch_loss = loss(preds, batch_y)
                batch_loss.backward()
                optimizer.step()

            # Compute epoch metrics
            with torch.no_grad():
                train_preds = self.model(X_train)
                val_preds = self.model(X_val)

                train_loss = loss(train_preds, y_train).item()
                val_loss = loss(val_preds, y_val).item()

                train_acc = (train_preds.argmax(dim=1) == y_train).float().mean().item()
                val_acc = (val_preds.argmax(dim=1) == y_val).float().mean().item()

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

            if self.verbose:
                print(
                    f"Epoch: {epoch}/{self.epochs} - "
                    f"Train Loss {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                    f"Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}"
                )

        self.train_losses_ = train_losses
        self.val_losses_ = val_losses
        self.train_accs_ = train_accs
        self.val_accs_ = val_accs
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels (in original encoding)
        """
        self.model.eval()
        X_array = X.values if hasattr(X, "values") else X
        X_tensor = torch.tensor(X_array, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1).numpy()

        # Convert back to original labels
        return self.label_encoder_.inverse_transform(predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        self.model.eval()
        X_array = X.values if hasattr(X, "values") else X
        X_tensor = torch.tensor(X_array, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X_tensor)
            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(logits, dim=1).numpy()

        return probabilities

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            "model": self.model,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "verbose": self.verbose,
            "validation_split": self.validation_split,
            "random_state": self.random_state,
            "class_weight": self.class_weight,
        }
        return params

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

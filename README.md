# pytorchers

PyTorch helpers for tabular datasets with sklearn compatibility, model inspection, and visualization tools.

## Key Features

- **Sklearn-compatible PyTorch regressors** - Drop-in replacement for sklearn estimators with full Pipeline, GridSearchCV, and cross-validation support
- **Flexible neural network architectures** - Easily configure feedforward networks for tabular data
- **Model inspection and visualization** - Visualize neuron activations, compare train/validation patterns, and understand model behavior
- **Training utilities** - Automatic train/validation splits, progress tracking, and loss history

**Python:** >=3.9.16

**Note:** Currently, the `y` (target) parameter in `fit()` expects a pandas Series or DataFrame with a `.values` attribute. Support for plain numpy arrays will be added in a future release.

## Installation

For local development:

```bash
# Clone the repository
git clone <repository-url>
cd pytorchers

# Install in editable mode
pip install -e .

# Or using uv (recommended)
uv pip install -e .
```

## Quick Start

```python
from pytorchers.reg import NNRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Prepare data (currently expects pandas DataFrames/Series for y)
# X can be numpy array or pandas DataFrame
# y should be pandas Series
X_train = ...  # Shape: (n_samples, n_features)
y_train = ...  # pandas Series with shape (n_samples,)

# Create a neural network regressor
model = NNRegressor(
    input_size=13,        # Number of input features
    layers=[64, 32],      # Hidden layer sizes
    epochs=100,
    batch_size=32,
    random_state=42
)

# Use in sklearn Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', model)
])

# Fit and predict
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

# Access training history
print(f"Final training loss: {model.train_losses_[-1]:.4f}")
print(f"Final validation loss: {model.val_losses_[-1]:.4f}")
```

## Core Features

### Sklearn-Compatible Regressors

#### NNRegressor - High-Level API

The simplest way to use PyTorch neural networks with sklearn:

```python
from pytorchers.reg import NNRegressor

model = NNRegressor(
    input_size=10,
    layers=[64, 32, 16],     # 3 hidden layers
    output_size=1,
    epochs=100,
    batch_size=32,
    lr=0.001,
    validation_split=0.2,
    verbose=True,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### NNRegressorEstimator - Wrapper for Custom Models

For more control, wrap your custom PyTorch models:

```python
from pytorchers.reg import NNRegressorEstimator, BaseNNRegressor

# Create custom PyTorch model
custom_model = BaseNNRegressor(
    input_size=10,
    layers=[128, 64, 32],
    output_size=1
)

# Wrap for sklearn compatibility
estimator = NNRegressorEstimator(
    model=custom_model,
    epochs=200,
    batch_size=64,
    lr=0.001,
    verbose=True
)

estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)
```

**Key Parameters:**
- `model`: PyTorch nn.Module instance
- `loss`: Loss function ("mse", "mae", "huber") - currently only MSE implemented
- `optimizer`: Optimizer type ("adam", "sgd", "rmsprop") - currently only Adam implemented
- `lr`: Learning rate (default: 0.001)
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size for training (default: 64)
- `validation_split`: Fraction of data for validation (default: 0.1)
- `random_state`: Seed for reproducibility

### Model Architecture

#### BaseNNRegressor

Customizable feedforward neural network for regression:

```python
from pytorchers.reg import BaseNNRegressor

model = BaseNNRegressor(
    input_size=13,
    layers=[64, 32, 8],   # 3 hidden layers with ReLU activations
    output_size=1         # Single regression output
)
```

The model automatically creates:
- Linear layers with specified sizes
- ReLU activations between layers
- Final output layer (no activation)

### Model Inspection & Visualization

#### ForwardTracker Mixin

Visualize and understand your neural network's internal behavior:

```python
import torch
import torch.nn as nn
from pytorchers.viz import ForwardTracker

# Create an inspectable model
class InspectableRegressor(nn.Module, ForwardTracker):
    def __init__(self, input_size, layers):
        nn.Module.__init__(self)
        ForwardTracker.__init__(self)

        # Define your architecture
        self.layers = nn.ModuleList()
        in_size = input_size
        for hidden_size in layers:
            self.layers.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size
        self.output = nn.Linear(in_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

# Create and train model
model = InspectableRegressor(13, [64, 32])
# ... training code ...

# Visualize neuron activations
model.plot_activations(
    X_tensor,
    y_tensor,
    agg_func=torch.mean,
    fig_title="Mean Neuron Activations"
)

# Compare train vs validation activations
model.plot_compared_activations(
    dataset1=(X_train_tensor, y_train_tensor),
    dataset2=(X_val_tensor, y_val_tensor),
    agg_func=torch.mean,
    fig_title1="Training Set",
    fig_title2="Validation Set"
)
```

The visualization shows:
- Heatmaps of neuron activations across layers
- Truth vs prediction scatter plots
- Side-by-side comparisons of different datasets

## Usage Examples

### Basic Regression

```python
from pytorchers.reg import NNRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = NNRegressor(
    input_size=X_train.shape[1],
    layers=[64, 32],
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    random_state=42
)

model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(f"MSE: {mean_squared_error(y_test, predictions):.4f}")
print(f"R2: {r2_score(y_test, predictions):.4f}")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', NNRegressor(input_size=13, layers=[64, 32], epochs=50))
])

# Cross-validation scores
cv_scores = cross_val_score(
    pipe, X_train, y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)
print(f"CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Cross-validation predictions
cv_predictions = cross_val_predict(pipe, X_train, y_train, cv=5)
```

### Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'regressor__layers': [[64, 32], [128, 64], [64, 32, 16]],
    'regressor__lr': [0.001, 0.01],
    'regressor__batch_size': [32, 64]
}

# Grid search
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', NNRegressor(input_size=13, epochs=50))
])

grid_search = GridSearchCV(
    pipe, param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {-grid_search.best_score_:.4f}")
```

### Model Inspection Workflow

```python
import torch
import torch.nn as nn
from pytorchers.viz import ForwardTracker

# Define inspectable model
class BostonRegressor(nn.Module, ForwardTracker):
    def __init__(self, input_size, layers):
        nn.Module.__init__(self)
        ForwardTracker.__init__(self)

        self.layers = nn.ModuleList()
        in_size = input_size
        for hidden_size in layers:
            self.layers.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size
        self.output = nn.Linear(in_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

# Train model
model = BostonRegressor(13, [64, 32])
# ... training code ...

# Convert data to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Visualize mean activations
model.plot_activations(
    X_train_tensor,
    y_train_tensor,
    agg_func=torch.mean,
    fig_title="Training Set - Mean Activations"
)

# Visualize activation variance (neuron sensitivity)
model.plot_activations(
    X_train_tensor,
    y_train_tensor,
    agg_func=torch.std,
    fig_title="Training Set - Activation Variance"
)

# Compare train vs validation
model.plot_compared_activations(
    dataset1=(X_train_tensor, y_train_tensor),
    dataset2=(X_val_tensor, y_val_tensor),
    agg_func=torch.mean,
    fig_title1="Training Set",
    fig_title2="Validation Set"
)
```

## API Reference

### NNRegressor

High-level sklearn-compatible neural network regressor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | Required | Number of input features |
| `layers` | list[int] | [32, 32, 8] | Hidden layer sizes |
| `output_size` | int | 1 | Number of output values |
| `loss` | str | "mse" | Loss function (only "mse" currently) |
| `optimizer` | str | "adam" | Optimizer (only "adam" currently) |
| `lr` | float | 0.001 | Learning rate |
| `epochs` | int | 100 | Number of training epochs |
| `batch_size` | int | 32 | Batch size for training |
| `shuffle` | bool | True | Shuffle data during training |
| `verbose` | bool | True | Print training progress |
| `validation_split` | float | 0.2 | Validation set fraction |
| `random_state` | int | None | Random seed for reproducibility |

**Attributes:**
- `train_losses_`: List of training losses per epoch
- `val_losses_`: List of validation losses per epoch
- `is_fitted_`: Boolean indicating if model is fitted

**Methods:**
- `fit(X, y)`: Train the model
- `predict(X)`: Generate predictions
- `get_params(deep=True)`: Get parameters for GridSearchCV
- `set_params(**params)`: Set parameters for GridSearchCV

### NNRegressorEstimator

Sklearn wrapper for custom PyTorch models.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | Required | PyTorch model instance |
| `loss` | str | "mse" | Loss function |
| `optimizer` | str | "adam" | Optimizer type |
| `lr` | float | 0.001 | Learning rate |
| `epochs` | int | 100 | Training epochs |
| `batch_size` | int | 64 | Batch size |
| `shuffle` | bool | True | Shuffle training data |
| `verbose` | bool | True | Print progress |
| `validation_split` | float | 0.1 | Validation fraction |
| `random_state` | int | None | Random seed |

### BaseNNRegressor

Customizable feedforward neural network.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | Required | Number of input features |
| `layers` | list[int] | [32, 32, 8] | Hidden layer sizes |
| `output_size` | int | 1 | Number of outputs |

### ForwardTracker

Mixin class for model inspection and visualization.

**Methods:**
- `forward_track()`: Register hooks to track activations
- `plot_activations(X, y, agg_func=torch.mean, fig_title="")`: Visualize layer activations
- `plot_compared_activations(dataset1, dataset2, agg_func, fig_title1, fig_title2)`: Compare activations between datasets

**Parameters for plot methods:**
- `X`: Input tensor
- `y`: Target tensor
- `agg_func`: Aggregation function (torch.mean, torch.std, etc.)
- `fig_title`: Title for the plot

## Examples & Notebooks

See `notebooks/boston.ipynb` for a comprehensive demonstration using the Boston Housing dataset, including:
- Exploratory data analysis
- Basic regression with sklearn Pipeline integration
- Cross-validation examples
- Model inspection and activation visualization
- Comparing train/validation patterns

## Roadmap

### Upcoming Features
- **Classification support** - Binary and multi-class classification with sklearn compatibility
- **Additional loss functions** - MAE, Huber loss implementations
- **Additional optimizers** - SGD, RMSprop support
- **Enhanced visualization** - More inspection tools and plotting options
- **Type hints** - Full type annotation coverage
- **Comprehensive testing** - Unit and integration tests

## Contributing & Development

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd pytorchers

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality Tools

The project uses:
- **Ruff** for linting and formatting (line length: 100)
- **Pre-commit hooks** for automated quality checks
- **nbstripout** to keep notebooks clean in git
- **detect-secrets** for security scanning

### Running Pre-commit Hooks

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files
pre-commit run
```

### Project Structure

```
pytorchers/
├── src/pytorchers/      # Main package
│   ├── base.py          # Neural network architectures
│   ├── reg.py           # Sklearn-compatible regressors
│   ├── viz.py           # Visualization tools
│   └── main.py          # CLI entry point
├── notebooks/           # Example notebooks
├── pyproject.toml       # Project configuration
└── .pre-commit-config.yaml
```

## License

[Add your license information here]

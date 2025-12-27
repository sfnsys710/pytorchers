# pytorchers

PyTorch helpers for tabular datasets with sklearn compatibility, model inspection, and visualization tools.

> **⚠️ Early Stage Development:** This package is in early development and was assembled from various notebook experiments. The API is not yet stable and you should expect breaking changes in future releases. Use in production at your own risk.

## Why pytorchers?

Deep learning is remarkably potent even for tabular data, but PyTorch's flexibility comes with overhead that's particularly repetitive for tabular use cases:

**Integration Challenges:**
- PyTorch models don't integrate natively with sklearn's ecosystem
- No built-in support for sklearn `Pipeline`, `GridSearchCV`, or cross-validation
- Can't leverage sklearn's rich set of preprocessing tools and model selection utilities

**Boilerplate Overhead:**
- Manual training loops for every experiment (data loading, batching, optimization, validation)
- This becomes repetitive quickly, especially when most tabular architectures are simple feedforward networks

**Architecture Patterns:**
- Tabular deep learning typically uses straightforward architectures (stacked linear layers with ReLU)
- Writing the same `nn.Module` structure repeatedly is tedious and error-prone

**pytorchers solves these problems** by providing sklearn-compatible wrappers, automated training loops, and reusable architecture templates - letting you focus on experimentation rather than boilerplate.

## Key Features

- **Sklearn-compatible PyTorch models** - Drop-in replacement for sklearn estimators with full Pipeline, GridSearchCV, and cross-validation support for both regression and classification
- **Flexible neural network architectures** - Easily configure feedforward networks for tabular data
- **Model inspection and visualization** - Visualize neuron activations, compare train/validation patterns, and understand model behavior
- **Training utilities** - Automatic train/validation splits, progress tracking, loss/accuracy history
- **Classification support** - Binary and multi-class classification with label encoding, class weighting, and probability predictions

**Python:** >=3.9.16

**Note:** For regression, the `y` (target) parameter in `fit()` currently expects a pandas Series or DataFrame with a `.values` attribute. Classification models support both pandas and numpy arrays. Full numpy support for regression will be added in a future release.

## Installation

> **Note:** This package is in early development. The API may change significantly between versions. Pin your version if using in any serious project.

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

**Not yet on PyPI** - This package is still experimental and not published to PyPI. Install from source only.

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

### Sklearn-Compatible Models

#### Regression

##### NNRegressor - High-Level Regression API

The simplest way to use PyTorch neural networks for regression:

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

##### NNRegressorEstimator - Wrapper for Custom Regression Models

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

#### Classification

##### NNClassifier - High-Level Classification API

PyTorch neural networks for binary and multi-class classification:

```python
from pytorchers.clf import NNClassifier
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Create classifier
model = NNClassifier(
    input_size=4,
    layers=[64, 32],
    n_classes=3,
    epochs=100,
    batch_size=32,
    lr=0.001,
    validation_split=0.2,
    random_state=42
)

model.fit(X, y)
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

##### NNClassifierEstimator - Wrapper for Custom Classification Models

Wrap custom PyTorch models for classification:

```python
from pytorchers.clf import NNClassifierEstimator, BaseNNClassifier

# Create custom model
custom_model = BaseNNClassifier(
    input_size=10,
    layers=[128, 64, 32],
    n_classes=2
)

# Wrap for sklearn compatibility
estimator = NNClassifierEstimator(
    model=custom_model,
    epochs=200,
    batch_size=64,
    lr=0.001,
    class_weight='balanced',  # Handle imbalanced classes
    verbose=True
)

estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)
```

**Key Parameters:**
- `model`: PyTorch nn.Module instance
- `loss`: Loss function ("crossentropy")
- `optimizer`: Optimizer type ("adam")
- `lr`: Learning rate (default: 0.001)
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size for training (default: 64)
- `validation_split`: Fraction of data for validation (default: 0.1)
- `class_weight`: 'balanced' or dict for handling imbalanced classes
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

#### BaseNNClassifier

Customizable feedforward neural network for classification:

```python
from pytorchers.clf import BaseNNClassifier

model = BaseNNClassifier(
    input_size=10,
    layers=[64, 32, 8],   # 3 hidden layers with ReLU activations
    n_classes=3           # Number of classes
)
```

The model automatically creates:
- Linear layers with specified sizes
- ReLU activations between layers
- Final output layer with n_classes outputs (logits, no softmax)

### Model Inspection & Visualization

#### Why Visualize Hidden Activations?

The `ForwardTracker` mixin is inspired by cutting-edge CNN visualization research, particularly the seminal work "Visualizing and Understanding Convolutional Neural Networks." That research showed how visualizing which parts of an image maximize hidden activations helped researchers understand that:

- **Early CNN layers** learn basic features (edges, corners, colors)
- **Later layers** learn higher-level semantic features (object parts, faces)

**Adapting this to tabular data:**

This visualization approach is particularly valuable for tabular datasets when you need to debug unexpected model behavior:

1. **Data Leakage Detection:**
   - Assumption: Similar predicted values should have similar hidden activation distributions
   - Strategy: Split your target into deciles and visualize train vs. test activations
   - Red flag: If distributions diverge significantly, you may have leakage or distribution shift

2. **Error Analysis:**
   - When the model makes large prediction errors (predicting too high or too low)
   - Compare the hidden activations of the error case against similar target values from training
   - Identify which layers show divergent activations - this reveals which features caused the model to deviate
   - Use this to decide whether to engineer features differently, add regularization, or investigate data quality

3. **Model Health Checks:**
   - Visualize activation patterns between train and validation sets
   - Ensure the model learns consistent representations across splits
   - Detect if certain neurons are "dead" (always near zero) or saturated

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

### Basic Classification

```python
from pytorchers.clf import NNClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = NNClassifier(
    input_size=X_train.shape[1],
    layers=[64, 32],
    n_classes=len(np.unique(y)),
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight='balanced',  # Handle imbalanced data
    random_state=42
)

model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions))
```

## API Reference

### Regression

#### NNRegressor

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

#### NNRegressorEstimator

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

#### BaseNNRegressor

Customizable feedforward neural network.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | Required | Number of input features |
| `layers` | list[int] | [32, 32, 8] | Hidden layer sizes |
| `output_size` | int | 1 | Number of outputs |

### Classification

#### NNClassifier

High-level sklearn-compatible neural network classifier.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | Required | Number of input features |
| `layers` | list[int] | [32, 32, 8] | Hidden layer sizes |
| `n_classes` | int | 2 | Number of classes |
| `loss` | str | "crossentropy" | Loss function |
| `optimizer` | str | "adam" | Optimizer |
| `lr` | float | 0.001 | Learning rate |
| `epochs` | int | 100 | Number of training epochs |
| `batch_size` | int | 32 | Batch size for training |
| `shuffle` | bool | True | Shuffle data during training |
| `verbose` | bool | True | Print training progress |
| `validation_split` | float | 0.2 | Validation set fraction |
| `class_weight` | str/dict | None | 'balanced' or custom weights |
| `random_state` | int | None | Random seed for reproducibility |

**Attributes:**
- `train_losses_`: List of training losses per epoch
- `val_losses_`: List of validation losses per epoch
- `train_accs_`: List of training accuracies per epoch
- `val_accs_`: List of validation accuracies per epoch
- `classes_`: Unique class labels
- `is_fitted_`: Boolean indicating if model is fitted

**Methods:**
- `fit(X, y)`: Train the model
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict class probabilities
- `get_params(deep=True)`: Get parameters for GridSearchCV
- `set_params(**params)`: Set parameters for GridSearchCV

#### NNClassifierEstimator

Sklearn wrapper for custom PyTorch classification models.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | Required | PyTorch model instance |
| `loss` | str | "crossentropy" | Loss function |
| `optimizer` | str | "adam" | Optimizer type |
| `lr` | float | 0.001 | Learning rate |
| `epochs` | int | 100 | Training epochs |
| `batch_size` | int | 64 | Batch size |
| `shuffle` | bool | True | Shuffle training data |
| `verbose` | bool | True | Print progress |
| `validation_split` | float | 0.1 | Validation fraction |
| `class_weight` | str/dict | None | Class weights |
| `random_state` | int | None | Random seed |

#### BaseNNClassifier

Customizable feedforward neural network for classification.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | Required | Number of input features |
| `layers` | list[int] | [32, 32, 8] | Hidden layer sizes |
| `n_classes` | int | 2 | Number of classes |

### Visualization

#### ForwardTracker

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

See the example notebooks for comprehensive demonstrations:

- `notebooks/reg_demo.ipynb` - Regression examples including:
  - Basic regression with sklearn Pipeline integration
  - Cross-validation and hyperparameter tuning
  - Model inspection and activation visualization
  - Comparing train/validation patterns

- `notebooks/clf_demo.ipynb` - Classification examples including:
  - Binary and multi-class classification
  - Handling imbalanced datasets with class weights
  - Probability predictions and decision boundaries
  - Label encoding for categorical targets

## Roadmap

### Known Limitations & Upcoming Features

This package was assembled from various notebook experiments and has several rough edges:

**High Priority Fixes:**
- **Numpy support for regression** - Fix y.values assumption in reg.py (classification already works)
- **API stabilization** - Settle on consistent parameter names and defaults
- **Comprehensive testing** - Unit and integration tests with sklearn estimator checks
- **Better error messages** - More helpful feedback when things go wrong

**Feature Enhancements:**
- **Additional loss functions** - MAE, Huber loss implementations for regression
- **Additional optimizers** - SGD, RMSprop, AdamW support
- **Enhanced visualization** - More inspection tools and plotting options
- **Early stopping** - Built-in patience-based early stopping
- **Learning rate scheduling** - Support for LR decay and warmup
- **Type hints** - Full type annotation coverage
- **Better documentation** - More examples and tutorials

**Long-term Ideas:**
- Support for categorical features (embeddings)
- Attention mechanisms for tabular data
- Integration with feature importance tools
- Model interpretation via SHAP/LIME

Contributions welcome! This is an experimental project and we're open to ideas.

## Contributing & Development

This project is in early stages and was born from consolidating useful patterns across multiple notebook experiments. We welcome contributions, especially:

- Bug reports and fixes
- Documentation improvements
- Additional tests
- Feature suggestions (open an issue first to discuss)
- Real-world use cases and feedback

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
│   ├── __init__.py      # Public API exports
│   ├── reg.py           # Regression models (BaseNNRegressor, NNRegressorEstimator)
│   ├── clf.py           # Classification models (BaseNNClassifier, NNClassifierEstimator)
│   ├── viz.py           # Visualization tools (ForwardTracker)
│   └── main.py          # CLI entry point
├── notebooks/           # Example notebooks
│   ├── reg_demo.ipynb   # Regression examples
│   └── clf_demo.ipynb   # Classification examples
├── pyproject.toml       # Project configuration
├── .pre-commit-config.yaml
└── README.md
```

## License

[Add your license information here]

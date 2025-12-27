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

# Create and use in sklearn Pipeline
model = NNRegressor(input_size=13, layers=[64, 32], epochs=100)
pipe = Pipeline([('scaler', StandardScaler()), ('regressor', model)])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

See the notebooks for detailed examples and tutorials.

## Core Features

### Sklearn-Compatible Models

#### Regression

##### NNRegressor - High-Level Regression API

The simplest way to use PyTorch neural networks for regression:

```python
from pytorchers.reg import NNRegressor

model = NNRegressor(input_size=10, layers=[64, 32, 16], epochs=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

##### NNRegressorEstimator - Wrapper for Custom Regression Models

Wrap custom PyTorch models for sklearn compatibility:

```python
from pytorchers.reg import NNRegressorEstimator, BaseNNRegressor

custom_model = BaseNNRegressor(input_size=10, layers=[128, 64, 32])
estimator = NNRegressorEstimator(model=custom_model, epochs=200)
estimator.fit(X_train, y_train)
```

#### Classification

##### NNClassifier - High-Level Classification API

PyTorch neural networks for binary and multi-class classification:

```python
from pytorchers.clf import NNClassifier

model = NNClassifier(input_size=4, layers=[64, 32], n_classes=3, epochs=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

##### NNClassifierEstimator - Wrapper for Custom Classification Models

Wrap custom PyTorch models for sklearn compatibility:

```python
from pytorchers.clf import NNClassifierEstimator, BaseNNClassifier

custom_model = BaseNNClassifier(input_size=10, layers=[128, 64, 32], n_classes=2)
estimator = NNClassifierEstimator(model=custom_model, class_weight='balanced')
estimator.fit(X_train, y_train)
```

### Model Architecture

#### BaseNNRegressor

Customizable feedforward neural network for regression. Automatically creates linear layers with ReLU activations and a final output layer (no activation).

```python
from pytorchers.reg import BaseNNRegressor
model = BaseNNRegressor(input_size=13, layers=[64, 32, 8], output_size=1)
```

#### BaseNNClassifier

Customizable feedforward neural network for classification. Automatically creates linear layers with ReLU activations and outputs raw logits (no softmax).

```python
from pytorchers.clf import BaseNNClassifier
model = BaseNNClassifier(input_size=10, layers=[64, 32, 8], n_classes=3)
```

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

Add visualization capabilities to any PyTorch model via multiple inheritance:

```python
import torch.nn as nn
from pytorchers.viz import ForwardTracker

class InspectableModel(nn.Module, ForwardTracker):
    def __init__(self, input_size, layers):
        nn.Module.__init__(self)
        ForwardTracker.__init__(self)
        # ... define your layers ...

    def forward(self, x):
        # ... your forward pass ...
        return x

# Visualize activations
model.plot_activations(X_tensor, y_tensor, agg_func=torch.mean)
model.plot_compared_activations(dataset1=(X_train, y_train), dataset2=(X_val, y_val))
```

Creates heatmaps of neuron activations and truth vs prediction plots. See notebooks for detailed examples.

## Usage Examples

All models work seamlessly with sklearn's Pipeline, cross-validation, and GridSearchCV:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', NNRegressor(input_size=13, layers=[64, 32]))
])
pipe.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)

# Hyperparameter tuning
param_grid = {'regressor__layers': [[64, 32], [128, 64]], 'regressor__lr': [0.001, 0.01]}
grid_search = GridSearchCV(pipe, param_grid, cv=3)
grid_search.fit(X_train, y_train)
```

**For complete examples**, see the notebooks which demonstrate:
- Regression: Boston Housing dataset with full workflow
- Classification: Iris dataset with class weighting and probability predictions
- Visualization: Activation analysis and error debugging

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

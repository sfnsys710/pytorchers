# tabularpytorchers

**Higher-level PyTorch abstractions** for tabular data with **visual inspection of neural nets**.

Stop writing boilerplate training loops. Start understanding what your models are learning.

> **⚠️ Early Stage Development:** This package is in early development and was assembled from various notebook experiments. The API is not yet stable and you should expect breaking changes in future releases. Use in production at your own risk.

## Why tabularpytorchers?

PyTorch is powerful for tabular data, but using it means writing the same boilerplate code repeatedly and flying blind when models misbehave. tabularpytorchers solves both problems:

### 1. Abstraction over PyTorch Boilerplate

**The Problem:**
- Every PyTorch experiment requires manual training loops (batching, optimization, validation splits)
- Sklearn integration requires custom wrapper classes every time
- Simple feedforward architectures require writing the same `nn.Module` structure repeatedly

**The Solution:**
- **Pre-built sklearn-compatible estimators** - Drop your PyTorch model into `NNRegressorEstimator` or `NNClassifierEstimator` and get automatic training loops, validation splits, and sklearn compatibility
- **Configurable architectures** - Use `BaseNNRegressor` or `BaseNNClassifier` instead of writing `nn.Module` classes for standard feedforward networks
- **Full sklearn integration** - Works seamlessly with `Pipeline`, `GridSearchCV`, and `cross_val_score` out of the box

### 2. Visual Inspection of Neural Nets

**The Problem:**
- When your tabular model makes bad predictions, you're flying blind
- Hard to tell if you have data leakage, distribution shift, or feature engineering issues
- No easy way to understand which features or layers are causing problems

**The Solution:**
- **ForwardTracker mixin** - Capture layer activations with a simple mixin class
- **Activation visualization** - Heatmaps showing what each layer learns across your dataset
- **Train/validation comparison** - Detect data leakage and distribution shift by comparing activation patterns
- **Error analysis** - Identify which layers behave differently for error cases vs correct predictions

## Key Features

**Abstraction over PyTorch:**
- **Sklearn-compatible wrappers** - `NNRegressorEstimator` and `NNClassifierEstimator` handle training loops, validation splits, and data conversions automatically
- **Pre-built architectures** - `BaseNNRegressor` and `BaseNNClassifier` eliminate repetitive `nn.Module` boilerplate
- **Full sklearn integration** - Works with `Pipeline`, `GridSearchCV`, and `cross_val_score` out of the box
- **Smart defaults** - Automatic label encoding, stratified splits for classification, progress tracking

**Visual Inspection of Neural Nets:**
- **ForwardTracker mixin** - Add activation tracking to any PyTorch model via multiple inheritance
- **Activation visualization** - Heatmaps showing what each layer learns
- **Comparative analysis** - Side-by-side train vs validation activation patterns
- **Debugging tools** - Detect data leakage, distribution shift, and problematic features

**Python:** >=3.9.16

**Note:** For regression, the `y` (target) parameter in `fit()` currently expects a pandas Series or DataFrame with a `.values` attribute. Classification models support both pandas and numpy arrays. Full numpy support for regression will be added in a future release.

## Installation

> **Note:** This package is in early development. The API may change significantly between versions. Pin your version if using in any serious project.

For local development:

```bash
# Clone the repository
git clone <repository-url>
cd tabularpytorchers

# Install in editable mode
pip install -e .

# Or using uv (recommended)
uv pip install -e .
```

**Not yet on PyPI** - This package is still experimental and not published to PyPI. Install from source only.

## Quick Start

### Abstraction: Sklearn-Compatible PyTorch in 3 Lines

```python
from tabularpytorchers.reg import BaseNNRegressor, NNRegressorEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Configure architecture (no need to write nn.Module class)
model = BaseNNRegressor(input_size=13, layers=[64, 32])

# 2. Wrap with estimator (automatic training loop, validation, sklearn compatibility)
estimator = NNRegressorEstimator(model=model, epochs=100)

# 3. Use like any sklearn model
pipe = Pipeline([('scaler', StandardScaler()), ('regressor', estimator)])
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

### Visual Inspection: Understand What Your Model Learns

```python
from tabularpytorchers.clf import BaseNNClassifier, NNClassifierEstimator
from tabularpytorchers.viz import ForwardTracker
import torch.nn as nn

# Add inspection to any model via mixin
class InspectableNet(nn.Module, ForwardTracker):
    def __init__(self, input_size, layers, n_classes):
        nn.Module.__init__(self)
        ForwardTracker.__init__(self)
        # ... define layers ...

    def forward(self, x):
        # ... forward pass ...
        return x

# Visualize activations and compare train vs validation
model = InspectableNet(input_size=10, layers=[64, 32], n_classes=2)
model.plot_compared_activations(
    dataset1=(X_train, y_train),
    dataset2=(X_val, y_val)
)
# Instantly see if distributions diverge (data leakage, distribution shift)
```

See the notebooks for detailed examples and tutorials.

## Core Features

### Sklearn-Compatible Models

#### Regression

##### NNRegressorEstimator - Wrapper for Custom Regression Models

Wrap custom PyTorch models for sklearn compatibility:

```python
from tabularpytorchers.reg import NNRegressorEstimator, BaseNNRegressor

custom_model = BaseNNRegressor(input_size=10, layers=[128, 64, 32])
estimator = NNRegressorEstimator(model=custom_model, epochs=200)
estimator.fit(X_train, y_train)
```

#### Classification

##### NNClassifierEstimator - Wrapper for Custom Classification Models

Wrap custom PyTorch models for sklearn compatibility:

```python
from tabularpytorchers.clf import NNClassifierEstimator, BaseNNClassifier

custom_model = BaseNNClassifier(input_size=10, layers=[128, 64, 32], n_classes=2)
estimator = NNClassifierEstimator(model=custom_model, class_weight='balanced')
estimator.fit(X_train, y_train)
```

### Model Architecture

#### BaseNNRegressor

Customizable feedforward neural network for regression. Automatically creates linear layers with ReLU activations and a final output layer (no activation).

```python
from tabularpytorchers.reg import BaseNNRegressor
model = BaseNNRegressor(input_size=13, layers=[64, 32, 8], output_size=1)
```

#### BaseNNClassifier

Customizable feedforward neural network for classification. Automatically creates linear layers with ReLU activations and outputs raw logits (no softmax).

```python
from tabularpytorchers.clf import BaseNNClassifier
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
from tabularpytorchers.viz import ForwardTracker

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
from tabularpytorchers.reg import BaseNNRegressor, NNRegressorEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV

# Create model and estimator
model = BaseNNRegressor(input_size=13, layers=[64, 32])
estimator = NNRegressorEstimator(model=model, epochs=100)

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', estimator)
])
pipe.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)

# Hyperparameter tuning
param_grid = {'regressor__epochs': [100, 200], 'regressor__lr': [0.001, 0.01]}
grid_search = GridSearchCV(pipe, param_grid, cv=3)
grid_search.fit(X_train, y_train)
```

**For complete examples**, see the notebooks which demonstrate:
- Regression: Boston Housing dataset with full workflow
- Classification: Iris dataset with class weighting and probability predictions
- Visualization: Activation analysis and error debugging

## API Reference

### Regression

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
cd tabularpytorchers

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
tabularpytorchers/
├── src/tabularpytorchers/      # Main package
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

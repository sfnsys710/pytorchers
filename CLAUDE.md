# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tabularpytorchers provides **higher-level PyTorch abstractions** for tabular data with **visual inspection of neural nets**. The library solves two core problems:

1. **Abstraction over PyTorch boilerplate** - Eliminates repetitive training loops, model building, and sklearn integration code for tabular use cases
2. **Visual inspection of neural nets** - Provides activation visualization tools to understand what your model is learning and debug unexpected behavior

**Core capabilities:**
- Sklearn-compatible wrappers that handle training loops, validation splits, and data conversions automatically
- Pre-built feedforward architectures with simple configuration (no need to write nn.Module classes repeatedly)
- ForwardTracker mixin for capturing and visualizing layer activations
- Full compatibility with sklearn Pipeline, GridSearchCV, and cross-validation

**⚠️ Early Stage:** This package was assembled from various notebook experiments and is in early development. The API is not yet stable - expect rough edges, missing features, and potential breaking changes. The codebase prioritizes experimentation over production readiness.

**Critical Constraint:** The `fit()` method in reg.py (lines 114, 116) expects `y` to have a `.values` attribute (pandas Series/DataFrame). This will fail with plain numpy arrays and needs fixing. Classification (clf.py) properly handles both pandas and numpy using `hasattr(y, 'values')` checks.

## Development Commands

### Environment Setup
```bash
# Install in editable mode (recommended: use uv)
uv pip install -e .

# Install with dev dependencies for notebooks
uv pip install -e ".[dev]"

# Install with pre-commit dependencies
uv pip install -e . --group pre-commit

# Install pre-commit hooks
pre-commit install
```

### Code Quality
```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run ruff linting (auto-fix)
ruff check --fix .

# Run ruff formatting
ruff format .

# Run on notebooks
nbqa ruff notebooks/ --fix
```

**Ruff Config:** Line length 100, target Python 3.12, includes pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade.

**Pre-commit Config Highlights:**
- nbstripout keeps outputs but strips execution metadata from notebooks
- detect-secrets excludes .ipynb files and package-lock.json
- nbqa-ruff applies ruff checks to notebooks

### No Tests Yet
There is currently no test suite. When adding tests, use pytest and ensure sklearn compatibility with `sklearn.utils.estimator_checks.check_estimator()`.

## Architecture

The library implements two core themes through complementary design patterns:

### 1. Abstraction over PyTorch - Wrapper Pattern (reg.py and clf.py)

**Goal:** Eliminate boilerplate training loops and sklearn integration code.

**Implementation:**
- `NNRegressorEstimator` and `NNClassifierEstimator` wrap any PyTorch nn.Module
- Implement sklearn base classes (`BaseEstimator` with `RegressorMixin`/`ClassifierMixin`)
- **Automate repetitive tasks:**
  - Training loop with batching and optimization
  - Automatic train/validation splits
  - Numpy ↔ tensor conversions
  - Loss tracking and progress reporting
  - Label encoding for classification (via `LabelEncoder`)
- **Pre-built architectures:**
  - `BaseNNRegressor` and `BaseNNClassifier` provide configurable feedforward networks
  - Users configure with simple parameters instead of writing nn.Module classes
  - Dynamic layer creation using `setattr`/`getattr` (not ModuleList)

### 2. Visual Inspection of Neural Nets - Mixin Pattern (viz.py)

**Goal:** Understand what the model is learning and debug unexpected behavior.

**Implementation:**
- `ForwardTracker` mixin adds activation visualization via multiple inheritance
- Usage: `class MyModel(nn.Module, ForwardTracker)`
- **Inspection capabilities:**
  - Uses PyTorch forward hooks to capture layer activations during forward passes
  - Visualize activation distributions across layers
  - Compare train vs validation activation patterns (detect data leakage, distribution shift)
  - Analyze activations for error cases to identify problematic features
- **Current limitation:** Only tracks direct child Linear layers (uses `named_children()`, not `named_modules()`)

### Critical Implementation Details

#### BaseNNRegressor and BaseNNClassifier (reg.py and clf.py)
Both use `setattr`/`getattr` for dynamic layer creation, NOT ModuleList:
```python
# Layers created as: self.fc0, self.fc1, self.fc2, ...
setattr(self, f"fc{i}", nn.Linear(...))

# Accessed in forward() as:
x = getattr(self, f"fc{i}")(x)
```

**Key difference:**
- `BaseNNRegressor` outputs single value (or `output_size` values)
- `BaseNNClassifier` outputs `n_classes` logits (no softmax in forward - handled by CrossEntropyLoss)

#### NNRegressorEstimator.fit() (reg.py)
Loss and optimizer are hardcoded inline (lines 102-105), not in helper methods:
```python
if self.loss == "mse":
    loss = nn.MSELoss()
if self.optimizer == "adam":
    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
```

**TODO:** Only MSE loss and Adam optimizer are implemented despite documentation mentioning "mae", "huber", "sgd", "rmsprop".

#### NNClassifierEstimator.fit() (clf.py)
- Uses `LabelEncoder` to handle non-numeric class labels (lines 113-115)
- Stores `self.classes_` attribute (sklearn requirement for classifiers)
- Implements stratified train/test split to preserve class distribution (line 123)
- Supports class weighting for imbalanced datasets via `class_weight` parameter:
  - `class_weight="balanced"`: Auto-compute inverse frequency weights
  - `class_weight={0: 1.5, 1: 1.0}`: Custom class weights
- Tracks both loss and accuracy during training (unlike regression which only tracks loss)

#### ForwardTracker (viz.py)
- Uses `named_children()` so only tracks direct child layers, not nested modules
- Checks `isinstance(child, nn.Linear)` - won't work with Conv2d, LSTM, etc.
- Sets `self.forward_tracking` flag to prevent duplicate hook registration
- Stores activations with `{name}_activations` keys in `self.activations` dict

### sklearn Compatibility Requirements

For GridSearchCV and Pipeline to work:
1. All `__init__` parameters must be stored as instance attributes (e.g., `self.epochs = epochs`)
2. `get_params(deep=True)` must return dict of all parameters
3. `set_params(**params)` must update attributes and return self
4. `fit()` must return self
5. `predict()` should use `check_is_fitted()` (currently missing in reg.py)

**Important:** `NNRegressorEstimator` expects a model instance to be passed in, allowing the same model to be used across cross-validation folds.

## Key Gaps and Known Issues

1. **Data type assumption in reg.py (HIGH PRIORITY):** reg.py lines 114, 116 call `y.values` - fails on numpy arrays
   - Fix: Add `y.values if hasattr(y, 'values') else y` (like clf.py does correctly)
   - This affects `NNRegressorEstimator.fit()`
   - clf.py already handles this correctly (lines 114, 127-130, 220, 245)

2. **Loss/optimizer limited in regression:** Only MSE and Adam work despite accepting other strings
   - Extract inline code (lines 102-105) to helper methods
   - Add implementations for "mae", "huber" losses and "sgd", "rmsprop" optimizers
   - Classification only supports CrossEntropyLoss and Adam (but is documented correctly)

3. **No fitted check:** `predict()` doesn't verify model is fitted in either reg.py or clf.py
   - Add `from sklearn.utils.validation import check_is_fitted`
   - Call `check_is_fitted(self, ['model'])` at start of `predict()` and `predict_proba()`

4. **ForwardTracker limitation:** Only works with direct child Linear layers
   - Consider using `named_modules()` for nested architectures
   - Won't work with models using nn.Sequential or deeply nested modules
   - Works well for BaseNNRegressor/BaseNNClassifier since they use direct children with setattr

## Project Structure

```
tabularpytorchers/
├── src/tabularpytorchers/
│   ├── __init__.py          # Public API exports (BaseNNRegressor, BaseNNClassifier, etc.)
│   ├── reg.py               # BaseNNRegressor + NNRegressorEstimator
│   ├── clf.py               # BaseNNClassifier + NNClassifierEstimator
│   ├── viz.py               # ForwardTracker mixin for visualization
│   └── main.py              # CLI entry point
├── notebooks/
│   ├── reg_demo.ipynb       # Regression examples
│   └── clf_demo.ipynb       # Classification examples
├── pyproject.toml           # uv-based build config
├── .pre-commit-config.yaml  # Ruff, nbqa, nbstripout, detect-secrets
├── README.md                # Comprehensive user documentation
└── CLAUDE.md                # This file
```

## Classification Implementation Notes

Classification support has been implemented in `clf.py`. Key features:

1. **BaseNNClassifier** - Similar to BaseNNRegressor but outputs `n_classes` logits
2. **NNClassifierEstimator** - Wraps PyTorch models with sklearn `ClassifierMixin`
3. **Key differences from regression:**
   - Uses `nn.CrossEntropyLoss()` (expects raw logits, no softmax in forward)
   - `predict()` returns class labels (uses argmax + inverse_transform)
   - `predict_proba()` returns class probabilities (applies softmax to logits)
   - Stores `classes_` attribute in `fit()` (sklearn requirement)
   - Uses `LabelEncoder` to handle string/categorical targets
   - Implements stratified train/test split
   - Supports class weighting for imbalanced data
   - Tracks both loss and accuracy during training

## Example Notebooks

- `notebooks/reg_demo.ipynb` - Regression examples demonstrating:
  - sklearn Pipeline integration
  - Cross-validation and GridSearchCV
  - Visualization with ForwardTracker
  - Train/validation comparison

- `notebooks/clf_demo.ipynb` - Classification examples demonstrating:
  - Binary and multi-class classification
  - Label encoding for categorical targets
  - Class weighting for imbalanced datasets
  - Probability predictions with `predict_proba()`

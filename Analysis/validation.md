# Validation Architecture 
**Source files**:
- sklearn/utils/validation.py (Main validation functions)
- sklearn/utils/_param_validation.py 
- sklearn/utils/_missing.py 
- sklearn/utils/_array_api.py 



## Summary

Every estimator calls validation functions before processing data to ensure:

- Data has the correct shape and type
- No missing or infinite values (unless explicitly allowed)
- Parameters are within valid ranges
- Estimators are fitted before prediction


## Core Validation Functions

### 1. check_array() 

This is the most frequently used validation function. It handles:

- Converting list/DataFrame to numpy array
- Checking for NaN/inf values
- Ensuring correct dimensionality
- Optional dtype conversion
- Sparse matrix handling

**Signature** :

```
def check_array(
    array,
    accept_sparse=False,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    ensure_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
    input_name="",
):
```

Example:

```
def check_array(array, ...):
    # Handle pandas extension dtypes
    if pandas_requires_conversion:
        array = array.astype(new_dtype)
    
    # Handle sparse matrices
    if sp.issparse(array):
        array = _ensure_sparse_format(array, accept_sparse, ...)
    
    # Handle dense arrays
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ComplexWarning)
            array = _asarray_with_order(array, order=order, xp=xp)
    
    # Check for NaN/inf
    _assert_all_finite(array, allow_nan=ensure_all_finite == "allow-nan")
    
    return array
```

Usage in Estimators:

``` 
class LogisticRegression:
    def fit(self, X, y):
        # Validate input features
        X = check_array(X, accept_sparse=['csr', 'csc'])
        # Validate target (handled separately)
        y = check_array(y, ensure_2d=False)
        # ... training code
```

### 2. check_X_y()

Validates both features and target together, ensuring:

- X and y have the same number of samples
- y is appropriate for the estimator type

Example:

``` 
def fit(self, X, y):
    # Validate both together
    X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
    # X and y now guaranteed to have same n_samples
```

### 3. check_is_fitted() 

Ensures that an estimator has been fitted before calling predict(), Preventing unfitted Predictions.

usage:

```
def predict(self, X):
    check_is_fitted(self, ['coef_', 'intercept_'])
    # Now safe to use these attributes
    return X @ self.coef_ + self.intercept_
```

how it works:

```
def check_is_fitted(estimator, attributes=None):
    if not hasattr(estimator, "fit"):
        raise TypeError("Estimator must implement fit method")
    
    if attributes is not None:
        for attr in attributes:
            if not hasattr(estimator, attr):
                raise NotFittedError(
                    f"This {estimator.__class__.__name__} instance is not fitted yet. "
                    f"Call 'fit' with appropriate arguments before using this method."
                )
```

### 4. check_consistent_length()

Ensures all input arrays have the same number of samples.

```
def check_consistent_length(*arrays):
    lengths = [_num_samples(X) for X in arrays if X is not None]
    if len(set(lengths)) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )
```

usage:

```
def fit(self, X, y, sample_weight=None):
    check_consistent_length(X, y, sample_weight)
    # All three now have same number of samples
```

### 5. check_random_state()

Converts various inputs into a valid RandomState instance.

```
def check_random_state(seed):
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(f"{seed} cannot be used to seed a numpy.random.RandomState instance")
```

### 6. assert_all_finite()

Checks if an array contains NaN or infinite values. It uses an O(n) time, O(1) space optimization.

```
def _assert_all_finite(X, allow_nan=False):
    # Fast path: try sum first (O(n), no extra memory)
    first_pass_isfinite = xp.isfinite(xp.sum(X))
    if first_pass_isfinite:
        return  # All good!
    
    # Slow path: detailed error message
    _assert_all_finite_element_wise(X, ...)
```

This optimization makes validation extremely fast for most cases.


## Parameter Validation

### The Constraint System:

Parameters are validated using a constraint-based system. Each estimator defines _parameter_constraints.

```
class LogisticRegression:
    _parameter_constraints = {
        "penalty": [StrOptions({"l1", "l2", "elasticnet", None})],
        "C": [Interval(Real, 0, None, closed="right")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],  # Special string
        "warm_start": ["boolean"],
    }
```

### The validate_params Decorator:

This decorator applies validation to functions:

```
@validate_params(
    {"X": ["array-like"], "y": ["array-like"]},
    prefer_skip_nested_validation=True
)
def some_function(X, y):
    # X and y are already validated
    pass
```

## Missing Value Detection

Two key utilities:

### is_scalar_nan()

```
def is_scalar_nan(x):
    return (
        not isinstance(x, numbers.Integral)
        and isinstance(x, numbers.Real)
        and math.isnan(x)
    )
```

np.isnan() wasn't used because it doesn't work on non-numeric types (e.g., strings) and can't handle float('nan').



### is_pandas_na()

```
def is_pandas_na(x):
    with suppress(ImportError):
        from pandas import NA
        return x is NA
    return False
```

## Array API Support

scikit-learn supports multiple array backends (NumPy, PyTorch, CuPy) through the Array API standard.

### get_namespace()

```
def get_namespace(*arrays):
    """Return the common array namespace."""
    if not array_api_dispatch:
        return np_compat, False
    
    namespace = array_api_compat.get_namespace(*arrays)
    return namespace, True
```

### device()

```
def device(*array_list):
    """Hardware device where the array data resides on."""
    # Returns device (CPU/GPU) or None for NumPy
```

This enables GPU support: estimators can detect if inputs are on GPU and keep computations there.


## Validation Flow in Estimators



```
fit() called
    ↓
check_X_y() / check_array()
    ↓
_assert_all_finite() (fast sum check)
    ↓
_ensure_sparse_format() (if sparse)
    ↓
_validate_params() (from base.py)
    ↓
Training code runs with validated data

predict() called
    ↓
check_is_fitted()
    ↓
check_array() (validate prediction input)
    ↓
Prediction code
```




















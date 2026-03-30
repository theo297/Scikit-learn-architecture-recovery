
# Design Patterns in scikit-learn

## 1. Introduction

Design patterns are reusable solutions to common problems in software design. scikit-learn makes extensive use of several classic design patterns to achieve its goals of consistency, extensibility, and ease of use. Understanding these patterns provides insight into how the library is structured and how to effectively extend it.

This document identifies and analyzes the key design patterns used throughout scikit-learn, showing where they appear in the codebase and explaining how they contribute to the overall architecture.

---

## 2. Strategy Pattern
### 2.1 Pattern Overview
|Attribute	| Description|
|-----------|-------------|
|Pattern Type	|Behavioral
|Intent	|Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
|Problem Solved|	Multiple algorithms for the same task with different tradeoffs; need to switch between them without modifying client code
Concrete Strategies:
### 2.2 Implementation in scikit-learn

The Strategy pattern is fundamental to scikit-learn. Every estimator implements the same interface, making algorithms interchangeable.

**The Strategy Interface:**

The strategy interface is defined through BaseEstimator in sklearn/base.py. While BaseEstimator does not define abstract fit/predict methods, it provides the parameter management infrastructure that all strategies share.

Code Reference: sklearn/base.py lines 67-260 - BaseEstimator class

```python

# sklearn/base.py - BaseEstimator provides parameter management
class BaseEstimator:
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        # Implementation at line 219
        pass
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        # Implementation at line 244
        pass
````

**Concrete Strategies:**

Each algorithm is a concrete strategy that implements the fit and predict methods. These are located in their respective module directories.

Code References:

    LogisticRegression: sklearn/linear_model/_logistic.py - full class definition

    RandomForestClassifier: sklearn/ensemble/_forest.py - full class definition

    SVC: sklearn/svm/_classes.py - full class definition

```python

# Example: LogisticRegression inherits BaseEstimator and implements fit/predict
# File: sklearn/linear_model/_logistic.py
class LogisticRegression(LinearClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        # Logistic regression specific fitting logic
        # Uses gradient descent or other optimization
        return self
    
    def predict(self, X):
        # Make predictions using learned coefficients
        pass
````

**The Context:**

User code acts as the context, selecting and using strategies. Meta-estimators like GridSearchCV and Pipeline also act as contexts that work with any strategy.

Code Reference: sklearn/model_selection/_search.py - GridSearchCV works with any estimator

### 2.3 Code Evidence
|Component	|File Location |	Role|
|-----------|--------------|--------|
|Strategy Interface |	sklearn/base.py lines 67-260 |	BaseEstimator defines parameter contract|
|Concrete Strategies |	sklearn/linear_model/, sklearn/ensemble/, sklearn/svm/ |	Each algorithm implements fit/predict|
|Meta-estimators |	sklearn/pipeline.py, sklearn/model_selection/_search.py |	Contexts that work with any strategy|

### 2.4 Benefits of the Pattern

   - Interchangeability: Users can swap algorithms with minimal code changes
   - Encapsulation: Algorithm details are hidden behind the common interface
    -Extensibility: New algorithms can be added without modifying existing code
    -Testability: Each strategy can be tested independently
---

## 3. Composite Pattern

### 3.1 Pattern Overview

| Attribute | Description |
|-----------|-------------|
| **Pattern Type** | Structural |
| **Intent** | Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions uniformly. |
| **Problem Solved** | Need to treat a group of objects the same way as a single object; complex workflows need to be composed from simple steps |

### 3.2 Implementation in scikit-learn

The Pipeline class is the primary implementation of the Composite pattern. It composes multiple transformers and an estimator into a single object that itself behaves like an estimator.

**Component Interface:**
The common interface is defined by BaseEstimator and the mixin classes. All components must implement fit, and depending on their role, may implement predict or transform.

Code Reference: sklearn/base.py lines 275-297 - TransformerMixin, ClassifierMixin, RegressorMixin

**Leaf Components:**

Individual estimators and transformers are leaf nodes that do not contain other estimators.

Code References:

    -StandardScaler: sklearn/preprocessing/_data.py
    -LogisticRegression: sklearn/linear_model/_logistic.py
    
**Composite Component:**

Pipeline is the composite that contains and coordinates leaf components:

```# sklearn/pipeline.py - Pipeline composite implementation
class Pipeline(_BaseComposition):
    """Pipeline of transforms with a final estimator."""
    
    def __init__(self, steps, *, memory=None, verbose=False):
        self.steps = steps  # List of (name, estimator) tuples
    
    def fit(self, X, y=None, **fit_params):
        """Fit the pipeline.
        
        Applies transformers sequentially, then fits final estimator.
        """
        # Apply each transformer sequentially (implementation at line ~500)
        for name, transform in self._iter(with_final=False):
            X = transform.fit_transform(X, y)
        
        # Fit final estimator
        self._final_estimator.fit(X, y, **fit_params)
        return self
    
    def predict(self, X, **predict_params):
        """Predict using the pipeline.
        
        Applies transformers to data, then predicts with final estimator.
        """
        # Pass data through all transformers
        for name, transform in self._iter(with_final=False):
            X = transform.transform(X)
        
        # Predict using final estimator
        return self._final_estimator.predict(X, **predict_params)
```

**Client Usage:**

Clients treat leaf and composite objects identically because Pipeline implements the same interface.

Code Reference: Any estimator usage in scikit-learn examples shows this uniformity

### 3.3 Code Evidence

| Component | File Location | Role |
|-----------|---------------|------|
| Pipeline | sklearn/pipeline.py lines 434-700 | Composite implementation |
| _BaseComposition | sklearn/pipeline.py | Abstract base for composites |
| TransformerMixin | sklearn/base.py line 275 | Interface for transformer leaves |
| BaseEstimator | sklearn/base.py lines 67-260 | Interface for estimator leaves |

### 3.4 Benefits of the Pattern

- **Uniformity**: Pipelines and single estimators share the same interface
- **Simplicity**: Users don't need to know whether they are using a composite or leaf
- **Flexibility**: Workflows can be nested and composed arbitrarily
- **Maintainability**: Changes to the composite structure don't affect clients

---

## 4. Template Method Pattern

### 4.1 Pattern Overview

| Attribute | Description |
|-----------|-------------|
| **Pattern Type** | Behavioral |
| **Intent** | Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Subclasses can redefine certain steps without changing the algorithm's structure. |
| **Problem Solved** | Multiple algorithms share the same overall structure but differ in specific steps |

### 4.2 Implementation in scikit-learn

The Template Method pattern appears throughout scikit-learn, particularly in base classes that define the structure of fit and predict operations.

**Abstract Template:**

Base classes define the algorithm skeleton through methods like fit, which calls common validation steps before delegating to implementation-specific logic.

Code Reference: sklearn/base.py - BaseEstimator and base classes

```p# sklearn/base.py - Template method pattern in fit
class BaseEstimator:
    def fit(self, X, y=None):
        """Template method defining the fit skeleton."""
        # Common validation occurs via _validate_data
        # Implementation specific to subclass occurs in _fit method
        return self
    
    def _validate_data(self, X, y=None, **kwargs):
        """Common validation logic - called at start of fit."""
        # Implementation at line ~516
        # Used by all estimators
        pass
```

**Concrete Subclasses:**

Subclasses implement the algorithm-specific logic, often in private methods.

Code Reference: Various estimator files show this pattern

```# sklearn/linear_model/_logistic.py - LogisticRegression implementation
class LogisticRegression(LinearClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        # Template method: validation first
        X, y = self._validate_data(
            X, y, accept_sparse='csr', dtype=[np.float64, np.float32],
            order="C", accept_large_sparse=False
        )
        # Then algorithm-specific fitting
        return self._fit(X, y, sample_weight)
    
    def _fit(self, X, y, sample_weight):
        # Logistic regression specific implementation
        # Uses gradient descent or newton-cg optimization
        pass
```

### 4.3 Code Evidence

| Component | File Location | Template Method |
|-----------|---------------|-----------------|
| BaseEstimator | sklearn/base.py | fit method structure |
| BaseSVC | sklearn/svm/_base.py | fit method for SVMs |
| BaseForest | sklearn/ensemble/_forest.py | fit method for forests |
| _BaseComposition | sklearn/pipeline.py | fit and predict structures |

### 4.4 Benefits of the Pattern

- **Code Reuse**: Common steps are implemented once in the base class
- **Consistency**: All estimators follow the same validation and fitting pattern
- **Separation of Concerns**: The skeleton focuses on the algorithm flow; subclasses focus on implementation details
- **Extensibility**: New algorithms only need to implement the variable steps

---

## 5. Factory Pattern

### 5.1 Pattern Overview

| Attribute | Description |
|-----------|-------------|
| **Pattern Type** | Creational |
| **Intent** | Define an interface for creating an object, but let subclasses or helper functions decide which class to instantiate. |
| **Problem Solved** | Object creation is complex or requires configuration; need to simplify creation for common cases |

### 5.2 Implementation in scikit-learn

scikit-learn uses factory functions to simplify the creation of common objects, particularly pipelines.

**Factory Function:**

The make_pipeline function creates Pipeline objects with automatically generated step names.

Code Reference: sklearn/pipeline.py - make_pipeline function

```# sklearn/pipeline.py - make_pipeline factory function
def make_pipeline(*steps, memory=None, verbose=False):
    """Construct a Pipeline from the given estimators.
    
    This is a shorthand for the Pipeline constructor; it does not require,
    and does not permit, naming the estimators. Instead, their names will
    be set to the lowercase of their types automatically.
    
    Parameters
    ----------
    *steps : list of estimators
        A sequence of transformers and a final estimator.
    
    Returns
    -------
    p : Pipeline
        A pipeline with automatically generated step names.
    """
    # Generate step names from class names (actual implementation)
    # Returns Pipeline with auto-named steps
    pass
```

**Usage:**

```# Without factory: verbose - must name each step
pipeline = Pipeline([
    ('standardscaler', StandardScaler()),
    ('logisticregression', LogisticRegression())
])

# With factory: concise - names are auto-generated
pipeline = make_pipeline(StandardScaler(), LogisticRegression())
```

**Additional Factory Functions:**

Code References:

    -sklearn/pipeline.py - make_union for FeatureUnion
    -sklearn/compose/_column_transformer.py - make_column_transformer

### 5.3 Code Evidence

| Function | File Location | Purpose |
|----------|---------------|---------|
| make_pipeline | sklearn/pipeline.py | Create Pipeline with auto-named steps |
| make_union | sklearn/pipeline.py | Create FeatureUnion with auto-named transformers |
| make_column_transformer | sklearn/compose/_column_transformer.py | Create ColumnTransformer |

### 5.4 Benefits of the Pattern

- **Simplicity**: Reduces boilerplate code for common operations
- **Consistency**: Ensures objects are created with consistent naming and configuration
- **Readability**: Makes code more concise and expressive
- **Default Handling**: Provides sensible defaults for optional parameters

---

## 6. Adapter Pattern

### 6.1 Pattern Overview

| Attribute | Description |
|-----------|-------------|
| **Pattern Type** | Structural |
| **Intent** | Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces. |
| **Problem Solved** | Need to use existing functions or classes that don't implement the expected interface |

### 6.2 Implementation in scikit-learn

The FunctionTransformer class adapts arbitrary Python functions to the Transformer interface, allowing them to be used in pipelines.

**The Problem:**
Regular Python functions don't implement the fit and transform methods that transformers need for pipeline compatibility.

**The Adapter:**

FunctionTransformer adapts functions to the transformer interface:
Code Reference: sklearn/preprocessing/_function_transformer.py - FunctionTransformer class

```# sklearn/preprocessing/_function_transformer.py - FunctionTransformer adapter
class FunctionTransformer(TransformerMixin, BaseEstimator):
    """Constructs a transformer from an arbitrary callable."""
    
    def __init__(self, func=None, inverse_func=None, *, validate=False,
                 accept_sparse=False, check_inverse=True, feature_names_out=None):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        # Additional parameters for controlling behavior
    
    def fit(self, X, y=None):
        """Fit the transformer - does nothing for stateless functions."""
        if self.validate:
            # Validate input if requested
            X = check_array(X, accept_sparse=self.accept_sparse)
        return self
    
    def transform(self, X):
        """Apply the adapted function to the data."""
        if self.validate:
            X = check_array(X, accept_sparse=self.accept_sparse)
        
        if self.func is not None:
            # Delegate to the wrapped function
            return self.func(X)
        return X
    
    def inverse_transform(self, X):
        """Apply inverse transformation if provided."""
        if self.inverse_func is not None:
            return self.inverse_func(X)
        raise NotImplementedError("inverse_transform not implemented")
```

**Usage:**

```# The adapter makes any function pipeline-compatible
from sklearn.preprocessing import FunctionTransformer

def custom_preprocessing(X):
    return np.log(X + 1)

pipeline = Pipeline([
    ('custom', FunctionTransformer(custom_preprocessing)),
    ('model', LogisticRegression())
])

# Works with lambda functions too
pipeline = Pipeline([
    ('log', FunctionTransformer(lambda X: np.log(X + 1))),
    ('model', LogisticRegression())
])

# Even works with NumPy functions
pipeline = Pipeline([
    ('sqrt', FunctionTransformer(np.sqrt)),
    ('model', LogisticRegression())
])
```

### 6.3 Code Evidence

| Component | File Location | Role |
|-----------|---------------|------|
| FunctionTransformer | sklearn/preprocessing/_function_transformer.py | Adapter implementation |
| TransformerMixin | sklearn/base.py line 275 | Target interface |
| BaseEstimator | sklearn/base.py lines 67-260 | Required for pipeline compatibility |

### 6.4 Benefits of the Pattern

- **Reusability**: Existing functions can be used in pipelines without modification
- **Flexibility**: Any function can become a transformer
- **Simplicity**: No need to create custom transformer classes for simple operations
- **Interoperability**: Bridges the gap between simple functions and the transformer interface

---

## 7. Other Patterns in scikit-learn

### 7.1 Singleton Pattern

The global configuration system uses a singleton pattern to manage settings.
Code Reference: sklearn/_config.py - _Config class

```# sklearn/_config.py - Singleton for global configuration
class _Config:
    """Singleton holding global configuration."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

def get_config():
    """Return the global configuration singleton."""
    return _Config()
```

### 7.2 Decorator Pattern

Decorators add behavior to functions without modifying them.
Code Reference: sklearn/base.py - _fit_context decorator at line ~346

```# sklearn/base.py - Decorator for consistent fit context management
def _fit_context(func):
    """Decorator for consistent fit context management."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._config_context():
            return func(self, *args, **kwargs)
    return wrapper

# Used on all fit methods (see LogisticRegression.fit, etc.)
```

### 7.3 Proxy Pattern

Meta-estimators like GridSearchCV act as proxies to the real estimators.

Code Reference: sklearn/model_selection/_search.py - GridSearchCV class

```# sklearn/model_selection/_search.py - GridSearchCV as proxy
class GridSearchCV(BaseEstimator):
    def __init__(self, estimator, param_grid):
        self.estimator = estimator  # The real estimator being proxied
        self.param_grid = param_grid
    
    def fit(self, X, y):
        # Proxy handles cross-validation and parameter search
        # Delegates actual fitting to the wrapped estimator
        for params in self._param_combinations:
            estimator_clone = clone(self.estimator)
            estimator_clone.set_params(**params)
            estimator_clone.fit(X, y)
            self._scores.append(self._score(estimator_clone))
        return self
```

### 7.4 Builder Pattern

The Pipeline construction follows a builder-like pattern with methods for adding steps.

Code Reference: sklearn/pipeline.py - Pipeline class with set_params for nested configuration

---

## 8. Pattern Interactions

The patterns in scikit-learn work together to create a cohesive architecture:

<img width="607" height="888" alt="Screenshot 2026-03-30 193129" src="https://github.com/user-attachments/assets/e5379883-677d-48fe-ad35-4ec6bfbb8048" />
<img width="575" height="772" alt="Screenshot 2026-03-30 193135" src="https://github.com/user-attachments/assets/ef5a553d-1c68-4c73-b3e2-2708f1d8532d" />
<img width="664" height="1414" alt="Screenshot 2026-03-30 193141" src="https://github.com/user-attachments/assets/b8edc2e4-a186-4f97-b0be-d3e209bcedee" />

----

## 9. Why These Patterns Matter

Understanding these patterns provides several benefits:

**For Users:**
- Understanding the Strategy pattern explains why algorithms are interchangeable
- Knowing the Composite pattern explains why pipelines behave like single estimators

**For Developers:**
- The Template Method pattern shows where to add new algorithm variants
- The Adapter pattern shows how to integrate custom functions

**For Architects:**
- The pattern selection demonstrates tradeoffs between flexibility and complexity
- The combination of patterns shows how design principles like open-closed are implemented

---

## 10. Summary

scikit-learn employs a thoughtful combination of design patterns to achieve its goals:

| Pattern | Primary Role | Key Location |
|---------|--------------|--------------|
| Strategy | Algorithm interchangeability | All estimators |
| Composite | Workflow composition | Pipeline |
| Template Method | Consistent algorithm structure | BaseEstimator, base classes |
| Factory | Simplified object creation | make_pipeline |
| Adapter | Function integration | FunctionTransformer |

These patterns work together to create an architecture that is:

- **Consistent**: Same patterns repeated throughout the library
- **Extensible**: New components can be added by implementing existing patterns
- **Composable**: Patterns combine to enable complex workflows
- **Maintainable**: Each pattern encapsulates a specific concern

The patterns also align with the library's design principles: simplicity for users, flexibility for developers, and consistency across all components.

---

## 11. References

| Pattern | Primary File | Key Lines |
|---------|--------------|-----------|
|Strategy	|sklearn/base.py	|67-260
|Composite	sklearn/pipeline.py	|434-700
|Template Method|	sklearn/base.py	|67-260, 516
|Factory	|sklearn/pipeline.py	|make_pipeline function
|Adapter	|sklearn/preprocessing/_function_transformer.py	|Full class
|Decorator	|sklearn/base.py	|~346 (_fit_context)
|Proxy	|sklearn/model_selection/_search.py	|GridSearchCV class
|Singleton	|sklearn/_config.py	|_Config class




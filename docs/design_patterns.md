
# Design Patterns in scikit-learn

## 1. Introduction

Design patterns are reusable solutions to common problems in software design. scikit-learn makes extensive use of several classic design patterns to achieve its goals of consistency, extensibility, and ease of use. Understanding these patterns provides insight into how the library is structured and how to effectively extend it.

This document identifies and analyzes the key design patterns used throughout scikit-learn, showing where they appear in the codebase and explaining how they contribute to the overall architecture.

---

## 2. Strategy Pattern

### 2.1 Pattern Overview

| Attribute | Description |
|-----------|-------------|
| **Pattern Type** | Behavioral |
| **Intent** | Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it. |
| **Problem Solved** | Multiple algorithms for the same task with different tradeoffs; need to switch between them without modifying client code |

### 2.2 Implementation in scikit-learn

The Strategy pattern is fundamental to scikit-learn. Every estimator implements the same interface, making algorithms interchangeable.

**The Strategy Interface:**

All estimators implement the Estimator interface with at least the fit method:

```python
# The strategy interface is implicit through the BaseEstimator class
class BaseEstimator:
    def fit(self, X, y=None):
        """Learn from data - each strategy implements this differently"""
        raise NotImplementedError
```

**Concrete Strategies:**

Each algorithm is a concrete strategy that implements the interface:

```python
class LogisticRegression(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # Logistic regression specific fitting logic
        # Uses gradient descent or other optimization
        return self

class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # Build an ensemble of decision trees
        # Each tree trained on bootstrap sample
        return self

class SVC(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # Support vector machine optimization
        # Finds maximum margin hyperplane
        return self
```

**The Context:**

User code acts as the context, selecting and using strategies:

```python
# Context: User code chooses the strategy
class ClassificationTask:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def train(self, X, y):
        self.strategy.fit(X, y)
    
    def predict(self, X):
        return self.strategy.predict(X)

# Different strategies can be swapped in
strategies = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC()
]

for strategy in strategies:
    task = ClassificationTask(strategy)
    task.train(X_train, y_train)
    predictions = task.predict(X_test)
```

### 2.3 Code Evidence

| Component | File Location | Role |
|-----------|---------------|------|
| Strategy Interface | sklearn/base.py | BaseEstimator defines the contract |
| Concrete Strategies | sklearn/linear_model/, sklearn/ensemble/ | Each algorithm implements fit/predict |
| Meta-estimators | sklearn/pipeline.py, sklearn/model_selection/ | Contexts that work with any strategy |

### 2.4 Benefits of the Pattern

- **Interchangeability**: Users can swap algorithms with minimal code changes
- **Encapsulation**: Algorithm details are hidden behind the common interface
- **Extensibility**: New algorithms can be added without modifying existing code
- **Testability**: Each strategy can be tested independently

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

The common interface that both leaf and composite objects implement:

```python
class EstimatorInterface:
    def fit(self, X, y=None):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def transform(self, X):
        raise NotImplementedError
```

**Leaf Components:**

Individual estimators and transformers are leaf nodes:

```python
class StandardScaler(TransformerMixin, BaseEstimator):
    # Leaf: does not contain other estimators
    def fit(self, X, y=None):
        # Learn scaling parameters
        return self
    
    def transform(self, X):
        # Apply scaling
        return X_scaled

class LogisticRegression(BaseEstimator, ClassifierMixin):
    # Leaf: does not contain other estimators
    def fit(self, X, y=None):
        # Train model
        return self
    
    def predict(self, X):
        # Make predictions
        return predictions
```

**Composite Component:**

Pipeline is the composite that contains and coordinates leaf components:

```python
class Pipeline(_BaseComposition):
    def __init__(self, steps):
        self.steps = steps  # List of (name, estimator) tuples
    
    def fit(self, X, y=None):
        # Apply each transformer sequentially
        for name, transform in self.steps[:-1]:
            X = transform.fit_transform(X, y)
        
        # Fit final estimator
        self.steps[-1][1].fit(X, y)
        return self
    
    def predict(self, X):
        # Pass data through all transformers
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        
        # Predict using final estimator
        return self.steps[-1][1].predict(X)
```

**Client Usage:**

Clients treat leaf and composite objects identically:

```python
# Leaf: single estimator
simple_model = LogisticRegression()
simple_model.fit(X_train, y_train)
predictions = simple_model.predict(X_test)

# Composite: pipeline with multiple steps
complex_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('classifier', LogisticRegression())
])
complex_pipeline.fit(X_train, y_train)
predictions = complex_pipeline.predict(X_test)

# Both behave identically - same interface
```

### 3.3 Code Evidence

| Component | File Location | Role |
|-----------|---------------|------|
| Pipeline | sklearn/pipeline.py | Composite implementation |
| _BaseComposition | sklearn/pipeline.py | Abstract base for composites |
| TransformerMixin | sklearn/base.py | Interface for transformer leaves |
| BaseEstimator | sklearn/base.py | Interface for estimator leaves |

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

Base classes define the algorithm skeleton:

```python
class BaseEstimator:
    def fit(self, X, y=None):
        # Template method defining the skeleton
        self._validate_input(X, y)
        self._check_parameters()
        self._fit_implementation(X, y)
        self._post_fit_cleanup()
        return self
    
    def _validate_input(self, X, y):
        # Common validation logic - same for all subclasses
        X, y = check_X_y(X, y)
        return X, y
    
    def _check_parameters(self):
        # Common parameter validation
        pass
    
    def _fit_implementation(self, X, y):
        # Deferred to subclasses - the variable step
        raise NotImplementedError
    
    def _post_fit_cleanup(self):
        # Common cleanup - optional hook
        pass
```

**Concrete Subclasses:**

Subclasses implement the variable steps:

```python
class LogisticRegression(BaseEstimator):
    def _fit_implementation(self, X, y):
        # Implementation specific to logistic regression
        self.coef_ = self._optimize_loss(X, y)
        self.intercept_ = self._compute_intercept()
        return self
    
    def _optimize_loss(self, X, y):
        # Logistic regression specific optimization
        # Uses gradient descent or newton-cg
        pass

class Ridge(BaseEstimator):
    def _fit_implementation(self, X, y):
        # Implementation specific to ridge regression
        self.coef_ = self._solve_normal_equations(X, y)
        return self
    
    def _solve_normal_equations(self, X, y):
        # Ridge specific solution using linear algebra
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

The make_pipeline function creates Pipeline objects with a simplified interface:

```python
def make_pipeline(*steps):
    """Factory function for creating pipelines.
    
    Parameters
    ----------
    *steps : list of estimators
        A sequence of transformers and a final estimator.
    
    Returns
    -------
    pipeline : Pipeline
        A pipeline with automatically generated step names.
    """
    # Generate step names automatically
    names = []
    for i, step in enumerate(steps):
        name = step.__class__.__name__.lower()
        names.append(f"{name}-{i}")
    
    # Create and return the pipeline
    return Pipeline(list(zip(names, steps)))
```

**Usage:**

```python
# Without factory: verbose
pipeline = Pipeline([
    ('standardscaler', StandardScaler()),
    ('logisticregression', LogisticRegression())
])

# With factory: concise
pipeline = make_pipeline(StandardScaler(), LogisticRegression())
```

**Additional Factory Functions:**

```python
# Create a union of feature extractors
def make_union(*transformers):
    """Factory for FeatureUnion"""
    names = [f"{t.__class__.__name__.lower()}-{i}" for i, t in enumerate(transformers)]
    return FeatureUnion(list(zip(names, transformers)))

# Create a pipeline from a column transformer
def make_column_transformer(*transformers):
    """Factory for ColumnTransformer"""
    # Implementation details
    pass
```

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

Regular Python functions don't implement the fit and transform methods that transformers need:

```python
def custom_preprocessing(X):
    # This function doesn't have fit() or transform()
    return np.log(X + 1)

# Cannot use this directly in a pipeline
# pipeline = Pipeline([
#     ('custom', custom_preprocessing),  # This fails
#     ('model', LogisticRegression())
# ])
```

**The Adapter:**

FunctionTransformer adapts functions to the transformer interface:

```python
class FunctionTransformer(TransformerMixin, BaseEstimator):
    """Adapter that turns any function into a transformer."""
    
    def __init__(self, func=None, inverse_func=None, validate=False):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
    
    def fit(self, X, y=None):
        """Adapter implementation - does nothing since functions are stateless."""
        if self.validate:
            # Validate input if requested
            check_array(X)
        return self
    
    def transform(self, X):
        """Apply the adapted function to the data."""
        if self.validate:
            X = check_array(X)
        
        if self.func is not None:
            # Delegate to the wrapped function
            return self.func(X)
        return X
```

**Usage:**

```python
# The adapter makes any function pipeline-compatible
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
| TransformerMixin | sklearn/base.py | Target interface |
| BaseEstimator | sklearn/base.py | Required for pipeline compatibility |

### 6.4 Benefits of the Pattern

- **Reusability**: Existing functions can be used in pipelines without modification
- **Flexibility**: Any function can become a transformer
- **Simplicity**: No need to create custom transformer classes for simple operations
- **Interoperability**: Bridges the gap between simple functions and the transformer interface

---

## 7. Other Patterns in scikit-learn

### 7.1 Singleton Pattern

The global configuration system uses a singleton pattern to manage settings:

```python
# sklearn/_config.py
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

Decorators add behavior to functions without modifying them:

```python
# sklearn/base.py
def _fit_context(func):
    """Decorator for consistent fit context management."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._config_context():
            return func(self, *args, **kwargs)
    return wrapper

# Used on all fit methods
class LogisticRegression(BaseEstimator):
    @_fit_context
    def fit(self, X, y):
        # Fitting logic automatically gets context management
        pass
```

### 7.3 Proxy Pattern

Meta-estimators like GridSearchCV act as proxies to the real estimators:

```python
class GridSearchCV(BaseEstimator):
    def __init__(self, estimator, param_grid):
        self.estimator = estimator  # The real estimator
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

The Pipeline construction follows a builder-like pattern with methods for adding steps:

```python
# Using Pipeline with a builder-like pattern
pipeline = Pipeline([])
pipeline = pipeline.add_step('scaler', StandardScaler())
pipeline = pipeline.add_step('pca', PCA())
pipeline = pipeline.add_step('classifier', LogisticRegression())
```

---

## 8. Pattern Interactions

The patterns in scikit-learn work together to create a cohesive architecture:


**How Patterns Work Together:**

| Combination | How They Interact |
|-------------|-------------------|
| Strategy + Composite | Pipeline composites strategies; strategies remain interchangeable |
| Template Method + Strategy | Template defines structure; strategy provides implementation |
| Factory + Composite | Factory simplifies creation of composites |
| Adapter + Composite | Adapters allow non-strategies to become composite leaves |

---

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
| Strategy | sklearn/base.py | Full class definitions |
| Composite | sklearn/pipeline.py | Pipeline class, line 434-700 |
| Template Method | sklearn/base.py | BaseEstimator.fit structure |
| Factory | sklearn/pipeline.py | make_pipeline function |
| Adapter | sklearn/preprocessing/_function_transformer.py | FunctionTransformer class |
| Decorator | sklearn/base.py | _fit_context decorator |




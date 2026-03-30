# Recovered Architecture: scikit-learn

## 1. Architecture Recovery Method

This architecture was recovered using a combination of:A

- Static code analysis of the scikit-learn source code version 1.6.x
- Directory structure analysis to identify module organization
- Inheritance hierarchy tracing to understand class relationships
- Cross-referencing of key architectural components across the codebase

All claims in this document are supported by references to specific files, classes, and functions in the official scikit-learn GitHub repository: https://github.com/scikit-learn/scikit-learn

---

## 2. Architecture Overview

scikit-learn employs a layered architecture centered around a unified Estimator abstraction. The entire library is designed around three core principles:

- Consistent Interface: Every algorithm follows the same API pattern
- Composability: Components can be combined using meta-estimators
- Separation of Concerns: Validation, transformation, and modeling are distinct responsibilities

### 2.1 Core Abstraction: The Estimator

At the heart of scikit-learn lies the Estimator abstraction. Any object that can learn from data is an estimator. This abstraction is defined by:

- The fit(X, y) method which learns parameters from training data
- The ability to be combined with other estimators via meta-estimators

Code Evidence:

- sklearn/base.py defines BaseEstimator, the foundation class for all estimators
- sklearn/base.py defines ClassifierMixin and RegressorMixin which add behavioral contracts

### 2.2 Mapping to Course Concepts: Three Structure Categories

Following the framework from software architecture principles, scikit-learn's architecture comprises three categories of structures:

#### Module Structures (Implementation-Time)

| Structure Type | Location in scikit-learn | Purpose |
|----------------|--------------------------|---------|
| Decomposition | Directory hierarchy: sklearn/linear_model/, sklearn/ensemble/, etc. | Shows how system is broken into manageable modules |
| Uses | Import dependencies between modules | Determines which modules depend on others for correctness |
| Layer | API → Abstraction → Implementation → Utility | Enables portability and separation of concerns |
| Class | BaseEstimator inheritance hierarchy | Enables reuse through inheritance |

#### Component-and-Connector Structures (Runtime)

| Structure Type | Location in scikit-learn | Purpose |
|----------------|--------------------------|---------|
| Service | Pipeline, GridSearchCV | Components interact via fit/predict calls |
| Concurrency | Joblib-based parallelism via n_jobs parameter | Enables parallel execution where supported |

#### Allocation Structures (Mapping to Environment)

| Structure Type | Location in scikit-learn | Purpose |
|----------------|--------------------------|---------|
| Implementation | GitHub repository structure | Maps modules to file system |
| Deployment | CPU-only execution model | Maps components to hardware |
| Work Assignment | Open source contributor model (CONTRIBUTING.md) | Maps modules to development teams |

### 2.3 Architectural Views for Different Stakeholders

Different stakeholders need different views of the architecture to address their concerns:

| Stakeholder | View | Structure Shown | Purpose |
|-------------|------|-----------------|---------|
| Developer | Module View | Decomposition, Class | Understanding code organization |
| Data Scientist | Component View | Service | Understanding how to use the library |
| Maintainer | Layer View | Layer | Understanding dependencies |
| Contributor | Implementation View | Implementation, Work Assignment | Finding where to make changes |
| User | API View | Service | Learning the interface |
| System Architect | Quality View | All structures | Evaluating tradeoffs |

These views are represented in the diagrams included in this document.

### 2.4 Early Architectural Decisions

The following decisions were made early in scikit-learn's development and are now difficult to change:

| Decision | Impact | Why Hard to Change |
|----------|--------|---------------------|
| Estimator interface with fit/predict methods | All algorithms must conform | Thousands of users depend on this API |
| Base class inheritance pattern | All estimators inherit from BaseEstimator | Deeply embedded throughout the codebase |
| Python-first approach with Cython for performance | Performance-critical code in Cython | Would require complete rewrite for alternative approach |
| CPU-only execution model | No GPU acceleration | Architecture not designed for GPU from the start |
| NumPy and SciPy as core dependencies | Relies on these for data structures | Fundamental to how data is represented |

---

## 3. High-Level Architecture Diagram

<img width="815" height="1336" alt="Screenshot 2026-03-30 124847" src="https://github.com/user-attachments/assets/0ffe1864-5b4b-4771-99c8-c2c909802932" />

---

## 4. Main Architectural Layers

### 4.1 Estimator Abstraction Layer

**Reference:** sklearn/base.py

This layer defines the contract that all estimators must follow. The BaseEstimator class provides:

| Method | Purpose | Code Reference |
|--------|---------|----------------|
| get_params(deep=True) | Returns hyperparameters of the estimator | base.py line 219 |
| set_params(**params) | Sets hyperparameters with validation | base.py line 244 |

Key Insight: The get_params and set_params mechanism enables meta-estimators like GridSearchCV and Pipeline to work. Any estimator can be introspected and configured programmatically.

Mixin Classes:

- ClassifierMixin at line 291: Marker for classifiers, defines _estimator_type = classifier
- RegressorMixin at line 297: Marker for regressors, defines _estimator_type = regressor
- TransformerMixin at line 275: Adds fit_transform method

These mixins allow meta-estimators to determine at runtime whether they are working with a classifier, regressor, or transformer, enabling appropriate behavior.

### 4.2 Implementation Layer

**Reference:** sklearn/linear_model/, sklearn/ensemble/, sklearn/tree/

This layer contains concrete implementations of machine learning algorithms. Each implementation:

- Inherits from BaseEstimator
- Implements fit and optionally predict or transform
- May incorporate appropriate mixins such as ClassifierMixin for classifiers
- Calls validation utilities at the start of fit

Example from LogisticRegression in  sklearn/linear_model/_logistic.py :

```
class LogisticRegression(LinearClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        # Input validation happens first
        X, y = self._validate_data(
            X, y, accept_sparse='csr', dtype=[np.float64, np.float32],
            order="C", accept_large_sparse=False
        )
        # Then algorithm-specific fitting logic follows
```

### 4.3 Validation Layer

**Reference:** sklearn/utils/validation.py

This is not a separate physical layer but a cross-cutting concern invoked at the beginning of most fit methods. Key validation functions:

| Function | Purpose | Code Reference |
|----------|---------|----------------|
| check_array | Validates array format, dtype, dimensions | validation.py line 925 |
| check_X_y | Validates both features and labels together | validation.py line 1066 |
| check_is_fitted | Ensures estimator has been fitted before prediction | validation.py line 1247 |

How Validation Works:

- Called at the start of every estimator's fit method
- Ensures input data is in the correct format such as NumPy array or scipy sparse matrix
- Checks for NaN and infinite values which is configurable
- Validates dimensions match expectations

### 4.4 Meta-Estimator Layer

**Reference:** sklearn/pipeline.py, sklearn/model_selection/_search.py

This layer provides estimators that wrap and compose other estimators.

Pipeline in sklearn/pipeline.py :

- Implements the Composite pattern
- Chains multiple transformers and a final estimator
- Behaves like a single estimator with fit, predict, and score methods

Key implementation in pipeline.py at line 434:

```
class Pipeline(_BaseComposition):
    def fit(self, X, y=None, **fit_params):
        # Apply transformers sequentially
        for name, transform in self.steps[:-1]:
            X = transform.fit_transform(X, y)
        # Fit final estimator
        self.steps[-1][1].fit(X, y)
        return self
```

GridSearchCV in sklearn/model_selection/_search.py :

Uses get_params and set_params to explore hyperparameter combinations, demonstrating the power of the Estimator abstraction.

---

## 5. Core Components Deep Dive

### 5.1 BaseEstimator

**Reference:** sklearn/base.py lines 67-260

The BaseEstimator class provides two critical capabilities:

**Hyperparameter Management:**

- get_params recursively collects all parameters from an estimator and its components
- set_params allows setting parameters, even for nested components using double underscore syntax like estimator__param

**Validation of Inputs:**

- _validate_data method at line 516 centralizes input validation
- Called by most estimators at the start of fit

### 5.2 Mixin System

The mixin system uses multiple inheritance to add behavioral capabilities:

| Mixin | Purpose | Required Methods |
|-------|---------|------------------|
| ClassifierMixin | Marks a classifier, enables scoring | predict, score |
| RegressorMixin | Marks a regressor | predict, score |
| TransformerMixin | Adds fit_transform default | fit, transform |

Why mixins instead of a deep hierarchy:

- Allows flexible combination of behaviors
- A classifier that also transforms such as RandomTreesEmbedding can inherit both
- Avoids the diamond problem through careful design

### 5.3 Validation Utilities

**Reference:** sklearn/utils/validation.py

The validation system is invoked through three main functions. Here is the usage pattern across all estimators:

```python
def fit(self, X, y):
    # Validate input
    X, y = check_X_y(X, y, accept_sparse=True)
    
    # Check for NaN values if configured
    if self.force_all_finite:
        assert_all_finite(X)
    
    # Continue with fitting logic...
```

### 5.4 Pipeline Architecture

<img width="868" height="1137" alt="Screenshot 2026-03-30 124859" src="https://github.com/user-attachments/assets/bd6e1897-8778-4b11-a6d3-8be2ce7583ed" />

**Reference:** sklearn/pipeline.py

The Pipeline is a critical architectural component that enables workflow composition:


Pipeline Structure:



Key Implementation Details:

- Steps are stored as a list of name and estimator tuples
- All intermediate steps must implement fit and transform
- Final step must implement fit
- Pipeline itself implements fit, predict, and score, behaving as a single estimator

---

## 6. Key Design Patterns

| Pattern | Location | How It Is Used |
|---------|----------|----------------|
| Strategy | All estimators | Each algorithm is interchangeable via the same interface |
| Composite | pipeline.py | Pipeline composes multiple estimators into a single composite estimator |
| Template Method | fit methods | Base classes define algorithm skeleton; subclasses implement specific steps |
| Factory | make_pipeline | Convenience function for creating pipelines |
| Adapter | FunctionTransformer | Adapts arbitrary functions to the transformer interface |

### 6.1 Strategy Pattern in Detail

The Strategy pattern is fundamental to scikit-learn:

- Context: The user code calling fit and predict
- Strategy Interface: The Estimator interface with fit and predict methods
- Concrete Strategies: LogisticRegression, RandomForestClassifier, SVC, and others

All strategies are interchangeable as shown:

```python
strategies = [LogisticRegression(), RandomForestClassifier(), SVC()]
for strategy in strategies:
    strategy.fit(X_train, y_train)
    predictions = strategy.predict(X_test)
```

### 6.2 Composite Pattern in Pipeline

The Pipeline implements the Composite pattern:

- Component Interface: fit, predict, transform methods
- Leaf: Individual estimators and transformers
- Composite: Pipeline that contains multiple components and delegates to them

This allows clients to treat individual estimators and entire pipelines identically.

---

## 7. Component Relationships

The diagram below illustrates the inheritance and composition relationships:

<img width="865" height="1250" alt="Screenshot 2026-03-30 124908" src="https://github.com/user-attachments/assets/38f1bd63-231f-43d4-a827-c69228800c19" />

Key relationships:

- All estimators inherit directly or indirectly from BaseEstimator
- Behavioral capabilities are added via mixins using multiple inheritance
- Pipelines compose multiple estimators via composition, not inheritance
- Validation functions are called by estimators as dependencies, not through inheritance

---

## 8. Architectural Characteristics

### 8.1 Modularity

Each module such as linear_model, ensemble, and tree is largely independent. New algorithms can be added without modifying existing code.

Evidence: The sklearn directory contains over 30 top-level modules, each with clear responsibilities.

### 8.2 Consistency

The Estimator abstraction ensures that all models present the same interface to users.

Evidence: The _fit_context decorator in base.py at line 346 enforces consistent fitting behavior across all estimators.

### 8.3 Extensibility

New estimators can be created by:

- Inheriting from BaseEstimator
- Implementing fit and optionally predict or transform
- Adding appropriate mixins

Evidence: The third-party scikit-learn-contrib ecosystem demonstrates this extensibility with dozens of community-contributed estimators.

### 8.4 Separation of Concerns

- Validation: Handled by utils/validation.py
- Modeling: Implemented in algorithm-specific modules
- Composition: Handled by pipeline.py
- Model Selection: Handled by model_selection directory

---

## 9. Quality Attribute Analysis

Based on software architecture principles, scikit-learn's architecture enables or inhibits the following quality attributes:

### 9.1 Modifiability

Status: Enabled

How it works: Information hiding via BaseEstimator and separation of concerns make changes localized.

Evidence: Adding a new estimator requires only implementing fit and predict methods without touching existing code.

Code Reference: Any new estimator inherits from BaseEstimator in base.py.

### 9.2 Performance

Status: Partially Enabled

How it works: Cython for critical computational paths, joblib for parallelism where beneficial.

Tradeoff: Python overhead for flexibility versus raw execution speed.

Evidence: Cython files like `_tree.pyx` and the n_jobs parameter across many estimators.

### 9.3 Scalability

Status: Limited

How it works: Batch processing for large datasets, incremental learning via partial_fit where supported.

Limitation: Most algorithms require the full dataset to fit in memory.

Evidence: Only a subset of estimators implement the partial_fit method.

### 9.4 Reusability

Status: Enabled

How it works: Independent modules with clear interfaces can be extracted and reused.

Evidence: The linear_model module can be used independently of ensemble methods.

### 9.5 Security

Status: Partial

How it works: Input validation through check_array and check_X_y prevents injection attacks.

Limitation: No built-in encryption, access control, or audit logging.

Evidence: Validation functions in utils/validation.py provide the security boundary.

### 9.6 Availability

Status: Supported

How it works: Mature library with extensive testing and large user base.

Evidence: Comprehensive test suite in sklearn/tests directory.

### 9.7 Portability

Status: Enabled

How it works: Pure Python with Cython, minimal external dependencies beyond NumPy and SciPy.

Evidence: Works across Windows, Linux, and macOS without modification.

---

## 10. How the Architecture Aligns with Key Principles

| Principle | How scikit-learn Demonstrates It |
|-----------|----------------------------------|
| Architecture enables quality attributes | ^^ Section 9 for detailed analysis |
| Architecture allows reasoning about change | BaseEstimator isolates changes to single modules |
| Architecture enables early prediction of qualities | Can predict modifiability by examining BaseEstimator hierarchy |
| Documented architecture enhances communication | Unified API provides common language for all stakeholders |
| Architecture carries earliest design decisions | BaseEstimator interface was set early and rarely changes |
| Architecture defines implementation constraints | All estimators must implement fit |
| Architecture influences organizational structure | Open source teams organized by modules |
| Architecture enables evolutionary prototyping | Skeletal system possible with Pipeline infrastructure |
| Architecture improves estimates | Modular structure enables bottom-up estimation by teams |
| Architecture is transferable and reusable | Architecture reused across scikit-learn ecosystem |
| Architecture focuses on component assembly | Pipeline focuses on assembling existing components |
| Architecture restricts design vocabulary | Design constrained by Estimator pattern |
| Architecture is foundation for training | BaseEstimator serves as entry point for new developers |

---

## 11. Conway's Law: Organization Matches Architecture

As noted in software architecture literature, organizations design systems that mirror their communication structure. This is evident in scikit-learn:

- The module structure reflects the open source team organization
- Contributors specialize in specific modules such as linear_model or ensemble
- Communication channels like GitHub issues and pull requests align with module boundaries
- The architecture documentation matches how teams coordinate work

Evidence: The CONTRIBUTING.md file shows how the contributor community is organized around module boundaries.

---

## 12. Limitations of This Recovery

This architecture recovery represents the logical architecture rather than every physical implementation detail. Some complexities not captured include:

- Cython modules: Performance-critical code such as _tree.pyx is implemented in Cython and not visible in pure Python analysis
- Configuration flags: Some behavior changes based on global configuration settings
- Legacy code: Older parts of the library may not fully conform to patterns established in newer code
- Dynamic behavior: Some runtime behavior depends on data characteristics and cannot be fully captured through static analysis

---

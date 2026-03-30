# Pipeline Architecture 

**Source**:
- sklearn/pipeline/_pipeline.py 
- sklearn/compose/_column_transformer.py 
- sklearn/compose/_target.py 


## Summary

The pipeline module provides tools for connecting multiple estimators into a single, unified estimator. 

This enables:

- **Reproducibility**: All preprocessing steps and the final estimator are bundled together
- **Preventing data leakage**: Transformers are fitted only on training data, not the full dataset
- **Simplified cross-validation**: A single object can be passed to cross_val_score
- **Hyperparameter tuning**: Grid search over all pipeline parameters using `__` syntax




## The Pipeline Class

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Transformer
    ('svm', SVC())                     # Final estimator
])

# Use as a single estimator
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```


### Internal Structure
A Pipeline stores steps as a list of (name, estimator) tuples:


```
class Pipeline:
    def __init__(self, steps, *, memory=None, verbose=False):
        self.steps = steps
        self.memory = memory          # For caching
        self.verbose = verbose
    
    @property
    def named_steps(self):
        """Dictionary mapping step names to estimators."""
        return dict(self.steps)
```


### How Pipeline Works

***During fit():***

The pipeline propagates data through each transformer, then fits the final estimator:

```
def fit(self, X, y=None, **fit_params):
    X_transformed = X
    # Fit all transformers
    for name, transformer in self.steps[:-1]:
        if transformer is not None:
            X_transformed = transformer.fit_transform(X_transformed, y)
    # Fit final estimator
    self.steps[-1][1].fit(X_transformed, y, **fit_params)
    return self
```


***During predict():***

The pipeline transforms the input through all transformers, then predicts:

```
def predict(self, X):
    X_transformed = X
    for name, transformer in self.steps[:-1]:
        if transformer is not None:
            X_transformed = transformer.transform(X_transformed)
    return self.steps[-1][1].predict(X_transformed)
```


#### Data Flow Diagram 

```
Raw Data (X)
    ↓
┌─────────────────────────────────────┐
│ Step 1: StandardScaler              │
│  fit_transform() → learns mean, std │
│  transform() → applies scaling      │
└─────────────────────────────────────┘
    ↓ (scaled data)
┌─────────────────────────────────────┐
│ Step 2: PCA                         │
│  fit_transform() → learns components│
│  transform() → projects onto PCs    │
└─────────────────────────────────────┘
    ↓ (reduced data)
┌─────────────────────────────────────┐
│ Step 3: SVC                         │
│  fit() → learns decision boundary   │
│  predict() → class predictions      │
└─────────────────────────────────────┘
    ↓
Predictions
```



## Memory Caching
Pipelines can cache intermediate results to avoid recomputation when using the same pipeline multiple times.

```
from tempfile import mkdtemp
from shutil import rmtree

cachedir = mkdtemp()
pipeline = Pipeline(steps, memory=cachedir)
```


## FeatureUnion
FeatureUnion applies multiple transformers in parallel and concatenates their outputs.

```
from sklearn.pipeline import featureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

union = FeatureUnion([
    ('pca', PCA(n_components=3)),
    ('kbest', SelectKBest(k=5))
])

# Features from PCA and SelectKBest are concatenated
X_transformed = union.fit_transform(X)
```


### Data Flow Diagram

```
Input Data (X)
    ↓
    ├── PCA.transform() → features A (3 columns)
    │
    └── SelectKBest.transform() → features B (5 columns)
    ↓
Concatenated features (8 columns)
```


## Methods

| Method | Behavior |
| ------ | -------- |
|fit(X, y=None)	| Fits each transformer independently |
| transform(X)	| Transforms with each transformer, concatenates results |
|fit_transform(X, y=None) |	Fits and transforms all transformers, concatenates |




## ColumnTransformer

ColumnTransformer applies different transformers to different columns of the input.

```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('numeric', StandardScaler(), ['age', 'income']),
    ('categorical', OneHotEncoder(), ['gender', 'city']),
    ('passthrough', 'passthrough', ['id'])  # Pass through unchanged
])
# Can be used in a pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression())
])
```



#### Column Selection Methods
| Method | Description |
| ------ | ----------- |
|Column names |	['age', 'income'] — select specific columns by name |
|Column indices	| [0, 1] — select by integer position |
|Slice | slice(0, 3) — select a range |
|Boolean mask	| [True, False, True] — select by boolean array |
|Callable	| lambda X: X.columns[0:2] — dynamic selection |


### Transformer Types


| Value |	Meaning |
| ----- | ------- |
|Transformer | Any scikit-learn transformer |
|"passthrough"|	Pass columns through unchanged |
|"drop"|	Drop columns (exclude from output) |
|None	| Same as "drop" |


## TransformedTargetRegressor
TransformedTargetRegressor applies a transformation to the target variable before regression.

```
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer

model = TransformedTargetRegressor(
    regressor=LinearRegression(),
    transformer=QuantileTransformer(output_distribution='normal')
)

#Target is transformed before fitting, inverse-transformed after prediction
model.fit(X, y)
y_pred = model.predict(X)  # Returns predictions on original scale
```


## Parameter Naming Convention


Pipeline parameters use double underscore syntax to reference nested components:

```
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

#Access pipeline parameters
pipeline.get_params()
# Output includes:
#- 'scaler__with_mean': True
#- 'scaler__with_std': True
#- 'svm__C': 1.0
#- 'svm__kernel': 'rbf'

#Set parameters
pipeline.set_params(svm__C=10.0, scaler__with_mean=False)

# Grid search over pipeline parameters
param_grid = {
    'scaler__with_mean': [True, False],
    'svm__C': [0.1, 1.0, 10.0],
    'svm__kernel': ['linear', 'rbf']
}
```



### Nested Pipelines
For pipelines inside pipelines, use multiple underscores:

```
pipeline = Pipeline([
    ('preprocess', ColumnTransformer([
        ('numeric', Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA())
        ]), ['age', 'income'])
    ])),
    ('classifier', SVC())
])

# Access parameters
pipeline.get_params()
# Output includes:
# - 'preprocess__numeric__scaler__with_mean'
# - 'preprocess__numeric__pca__n_components'
# - 'classifier__C'
```


### Composite Pattern

Pipeline implements the Composite Pattern:

Component: BaseEstimator interface (fit, predict, transform)

Leaf: Individual estimators (StandardScaler, SVC, etc.)

Composite: Pipeline (treats sequence as single component)



## Special Methods

Pipelines also expose methods from the final estimator:

| Method | Behavior |
| ------ | -------- |
|predict_proba(X)| Calls predict_proba on final classifier |
|predict_log_proba(X) |	Calls predict_log_proba on final classifier |
|decision_function(X) | Calls decision_function on final estimator |
|score(X, y) | Calls score on final estimator |
|classes_	| Exposes final estimator's classes_ attribute |
|n_features_in_ |	Exposes number of features after all transformations |

If the final estimator doesn't have these methods, accessing them raises AttributeError.

## Utility Functions

***make_pipeline()***

Shorthand for creating pipelines without naming steps:

```
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), PCA(), SVC())
# Equivalent to:
# Pipeline([('standardscaler', StandardScaler()), ('pca', PCA()), ('svc', SVC())])
```

Step names are auto-generated from the transformer class names (lowercased).

***make_union()***

Shorthand for creating FeatureUnion:

```
from sklearn.pipeline import make_union

union = make_union(PCA(), SelectKBest())
```




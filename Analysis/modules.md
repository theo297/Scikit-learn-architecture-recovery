# Scikit-learn Module Architecture

**Source:** sklearn/
## Summary 

**functional modules** each handle a specific aspect of machine learning. 



## Module Classification

The modules can be classified as follows:

1. **Supervised Learning** — classification and regression algorithms
2. **Unsupervised Learning** — clustering, dimensionality reduction, manifold learning
3. **Data Transformation** — preprocessing, feature extraction, feature selection
4. **Model Selection & Evaluation** — validation, metrics, inspection
5. **Infrastructure** — utilities, testing, build tools


## Supervised Learning Modules

### linear_model/
**Purpose**: Linear models for regression and classification.

**Classes**:
- LinearRegression — ordinary least squares
- Ridge, RidgeCV — ridge regression with L2 regularization
- Lasso, LassoCV — Lasso with L1 regularization
- ElasticNet, ElasticNetCV — elastic net (L1 + L2)
- LogisticRegression — logistic regression classifier
- SGDClassifier, SGDRegressor — stochastic gradient descent

**Dependencies**: utils/, preprocessing/ (for scaling)


### tree/
**Purpose**: Decision tree algorithms for classification and regression.

**Classes**:
- DecisionTreeClassifier — classification tree
- DecisionTreeRegressor — regression tree
- ExtraTreeClassifier, ExtraTreeRegressor — extremely randomized trees

**Dependencies**: utils/



### ensemble/
**Purpose**: Ensemble methods that combine multiple base estimators.

**Classes**:
- RandomForestClassifier, RandomForestRegressor — bagged trees with random feature selection
- ExtraTreesClassifier, ExtraTreesRegressor — extremely randomized trees
- GradientBoostingClassifier, GradientBoostingRegressor — gradient boosting
- AdaBoostClassifier, AdaBoostRegressor — adaptive boosting
- BaggingClassifier, BaggingRegressor — bootstrap aggregating
- VotingClassifier, VotingRegressor — soft/hard voting
- StackingClassifier, StackingRegressor — stacked generalization
- HistGradientBoostingClassifier, HistGradientBoostingRegressor — histogram-based boosting (faster)

**Dependencies**: tree/, linear_model/, utils/



### svm/
**Purpose**: Support vector machines for classification, regression, and outlier detection.

**Classes**:
- SVC, NuSVC — support vector classification
- SVR, NuSVR — support vector regression
- LinearSVC, LinearSVR — linear SVM (uses liblinear)
- OneClassSVM — unsupervised outlier detection


**Dependencies**:utils/, preprocessing/


### neighbors/
**Purpose**: Nearest neighbor algorithms for classification, regression, and unsupervised learning.

**Classes**:
- KNeighborsClassifier, KNeighborsRegressor — k-nearest neighbors
- RadiusNeighborsClassifier, RadiusNeighborsRegressor — radius-based neighbors
- NearestNeighbors — unsupervised nearest neighbor search
- KernelDensity — kernel density estimation
- LocalOutlierFactor — outlier detection using LOF
- NeighborhoodComponentsAnalysis — metric learning

**Dependencies**: utils/, metrics/ (for distance metrics)


### neural_network/
**Purpose**: Multi-layer perceptron neural networks.

**Classes**:
- MLPClassifier — multi-layer perceptron classifier
- MLPRegressor — multi-layer perceptron regressor
-`BernoulliRBM — restricted Boltzmann machine


**Dependencies**: utils/, preprocessing/



### gaussian_process/
**Purpose**: Gaussian process models for regression and classification.

**Classes**:
- `GaussianProcessRegressor` — GP regression
- `GaussianProcessClassifier` — GP classification

**Dependencies**: utils/


### naive_bayes/
**Purpose**: Naive Bayes classifiers.

**Classes**:
- `GaussianNB` — Gaussian naive Bayes
- `MultinomialNB` — multinomial naive Bayes (for discrete counts)
- `BernoulliNB` — Bernoulli naive Bayes (for binary features)
- `ComplementNB` — complement naive Bayes

**Dependencies**: utils/


### discriminant_analysis/
**Purpose**: Linear and quadratic discriminant analysis.

**Classes**:
- LinearDiscriminantAnalysis — LDA (dimensionality reduction + classification)
- QuadraticDiscriminantAnalysis — QDA

**Dependencies**: utils/


## Unsupervised Learning Modules

### cluster/
**Purpose**: Clustering algorithms.

**Classes**:
- KMeans, MiniBatchKMeans — k-means clustering
- DBSCAN — density-based clustering
- OPTICS — ordering points to identify clustering structure
- AgglomerativeClustering — hierarchical clustering
- MeanShift — mean shift clustering
- SpectralClustering — spectral clustering
- BisectingKMeans — bisecting k-means
- AffinityPropagation — affinity propagation


**Dependencies**:utils/, metrics/ (for pairwise distances)


### decomposition/
**Purpose**: Dimensionality reduction and matrix decomposition.

**Classes**:
- PCA — principal component analysis
- TruncatedSVD — singular value decomposition (sparse-friendly)
- NMF — non-negative matrix factorization
- FactorAnalysis — factor analysis
- FastICA — independent component analysis
- IncrementalPCA — PCA for large datasets (online)
- SparsePCA, MiniBatchSparsePCA — sparse PCA
- KernelPCA — kernel PCA

**Dependencies**: utils/, preprocessing/


### manifold/
**Purpose**: Manifold learning for non-linear dimensionality reduction.

**Classes**:
- TSNE — t-distributed stochastic neighbor embedding
- MDS — multi-dimensional scaling
- Isomap — isometric mapping
- LocallyLinearEmbedding — locally linear embedding
- SpectralEmbedding — spectral embedding (Laplacian eigenmaps)

**Dependencies**: utils/, metrics/, neighbors/ (for graph construction)


### mixture/
**Purpose**: Gaussian mixture models.

**Classes**:
- GaussianMixture — expectation-maximization for GMMs
- BayesianGaussianMixture — variational inference for GMMs

**Dependencies**: utils/



### covariance/
**Purpose**: Covariance estimation.

**Classes**:
- EmpiricalCovariance — sample covariance
- ShrunkCovariance — shrunk covariance
- LedoitWolf — Ledoit-Wolf shrinkage
- OAS — oracle approximating shrinkage
- GraphicalLasso, GraphicalLassoCV — sparse inverse covariance


**Dependencies**: utils/



## Data Transformation Modules

### preprocessing/
**Purpose**: Data preprocessing and feature transformation.

**Classes**:
- StandardScaler — mean/variance scaling
- MinMaxScaler — feature scaling to [0, 1]
- MaxAbsScaler — scale by maximum absolute value
- RobustScaler — scale by percentile (robust to outliers)
- Normalizer — row-wise normalization
- OneHotEncoder — categorical encoding
- LabelEncoder — label encoding
- OrdinalEncoder — ordinal encoding
- TargetEncoder — target encoding
- KBinsDiscretizer — discretization into k bins
- PolynomialFeatures — polynomial feature generation
- SplineTransformer — B-spline basis functions
- FunctionTransformer — custom transformations


**Dependencies**: utils/



### feature_extraction/
**Purpose**: Feature extraction from unstructured data.

**Classes**:
- DictVectorizer — dict to feature matrix
- CountVectorizer — text to bag-of-words
- TfidfVectorizer — TF-IDF for text
- HashingVectorizer — feature hashing for large-scale text
- image.PatchExtractor — image patch extraction
- image.img_to_graph — image to graph


**Dependencies**: utils/, preprocessing/


### feature_selection/
**Purpose**: Feature selection methods.

**Classes**:
- SelectKBest — select top k features by score
- SelectPercentile — select features by percentile
- SelectFdr, SelectFwe, SelectFpr — statistical tests
- RFE, RFECV — recursive feature elimination
- SelectFromModel — selection based on model coefficients
- VarianceThreshold — remove constant/almost constant features
- SequentialFeatureSelector — forward/backward selection


**Dependencies**: utils/, base/



### impute/
**Purpose**: Missing value imputation.

**Classes**:
- SimpleImputer — mean/median/most_frequent/constant imputation
- KNNImputer — k-nearest neighbors imputation
- IterativeImputer — multivariate imputation by chained equations (MICE)

**Dependencies**: utils/, neighbors/ (for KNNImputer)


## Model Selection & Evaluation Modules

### model_selection/
**Purpose**: Model validation, hyperparameter tuning, and cross-validation.

**Classes**:
- train_test_split — train/test split
- cross_val_score, cross_val_predict, cross_validate — cross-validation
- GridSearchCV — exhaustive grid search
- RandomizedSearchCV — random parameter search
- HalvingGridSearchCV, HalvingRandomSearchCV — successive halving search
- ParameterGrid, ParameterSampler — parameter utilities
- KFold, StratifiedKFold, GroupKFold — cross-validation splitters
- LeaveOneOut, LeavePOut — leave-out cross-validation
- TimeSeriesSplit — time series CV



**Dependencies**: utils/, base/


### metrics/
**Purpose**: Scoring metrics for classification, regression, and clustering.

**Functions**:
- **Classification**: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
- **Regression**: mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
- **Clustering**: adjusted_rand_score, mutual_info_score, silhouette_score
- **Pairwise**: pairwise_distances, cosine_similarity, rbf_kernel

**Dependencies**: utils/


### inspection/
**Purpose**: Model inspection and interpretation tools.

**Functions**:
- partial_dependence, PartialDependenceDisplay — partial dependence plots
- permutation_importance — feature importance via permutation
- DecisionBoundaryDisplay — plot decision boundaries


**Dependencies**: utils/, metrics/, ensemble/


## Composition Modules


### compose/
**Purpose**: Meta-estimators for building complex workflows.

**Classes**:
- ColumnTransformer — apply different transformations to different columns
- TransformedTargetRegressor — transform target before regression
- make_column_transformer — helper function
- make_pipeline — helper for creating pipelines

**Dependencies**: utils/, pipeline/



### pipeline/
**Purpose**: Pipeline composition (wrapper around `compose/` for backwards compatibility).

**Classes**:
- Pipeline — chain transformers and final estimator
- FeatureUnion — concatenate results of multiple transformers

**Dependencies**: utils/, base/



## Infrastructure Modules

### utils/
**Purpose**: Cross-cutting utilities used by all modules.

**Submodules**:
- validation.py — input validation (`check_array`, `check_X_y`, `check_is_fitted`)
- _param_validation.py — parameter constraint validation
- _tags.py — estimator tagging system
- _array_api.py — array API compatibility
- _set_output.py — output container configuration
- _repr_html/ — HTML representation for notebooks
- _pprint.py — pretty printing for estimators
- _available_if.py — conditional method decorator
- _missing.py — missing value utilities
- _metadata_requests.py — metadata routing
- sparsefuncs.py — sparse matrix utilities
- extmath.py — extended math utilities
- estimator_checks.py — test utilities for estimators

**Dependencies**: None (lowest level)



### datasets/
**Purpose**: Built-in datasets for examples and testing.

**Functions**:
- load_iris, load_digits, load_wine, load_breast_cancer — classification datasets
- load_diabetes, load_linnerud — regression datasets
- fetch_olivetti_faces, fetch_lfw_people — image datasets
- make_classification, make_regression, make_blobs — synthetic dataset generators



**Dependencies**: utils/, preprocessing/



### externals/
**Purpose**: External library wrappers (e.g., joblib, array_api_compat).

**Exports**:
- joblib — parallel processing (patched)
- array_api_compat — array API compatibility layer
- array_api_extra — additional array utilities

**Dependencies**: None


### tests/
**Purpose**: Test suite for all modules.

**Structure**:
- Mirrors the main package structure
- test_common.py — common estimator tests
- test_base.py — base class tests
- Module-specific test files



### _loss/
**Purpose**: Loss functions for gradient boosting and other algorithms.

**Key Classes**:
- LossFunction — base class
- HalfBinomialLoss, HalfMultinomialLoss, HalfPoissonLoss, HalfGammaLoss — half losses
- PinballLoss — quantile regression


**Dependencies**: utils/



## Module Dependency DIAGRAM:
```
                ┌───────────────┐
                │  externals/   │
                └──────┬────────┘
                       ↓
        ┌──────────────┴──────────────┐
        │                             │
 ┌───────────────┐         ┌───────────────┐
 │     base/     │◄───────►│     utils/    │
 └───────────────┘         └───────────────┘
        │                             │
        └──────────────┬──────────────┘
                       ↓
                ┌───────────────┐
                │    _loss/     │
                └──────┬────────┘
                       ↓
 ┌───────────────────────────────────────────────┐
 │ preprocessing/  feature_extraction/           │
 │ feature_selection/                           │
 └───────────────────────────────────────────────┘
                       ↓
 ┌───────────────────────────────────────────────┐
 │ linear_model/  tree/  ensemble/  svm/         │
 │ neighbors/  gaussian_process/                 │
 │ neural_network/  naive_bayes/  etc.           │
 └───────────────────────────────────────────────┘
                       ↓
 ┌───────────────────────────────────────────────┐
 │ pipeline/  compose/  impute/                  │
 └───────────────────────────────────────────────┘
                       ↓
 ┌───────────────────────────────────────────────┐
 │ model_selection/  metrics/  inspection/       │
 └───────────────────────────────────────────────┘
```



## Observations

1. **Layered Architecture**: utils/ is the foundation; estimators depend on it; meta-estimators compose estimators.

2. **Public API in __init__.py**: Each module's __init__.py carefully curates what is exposed to users.

3. **Internal Files**: Files prefixed with `_` (e.g., _base.py, _classes.py) are internal implementation details.

4. **Cython for Performance**: Performance-critical code (e.g., tree building, k-means) is implemented in `.pyx` files.

5. **Test Suite Mirroring**: tests/ mirrors the module structure, making testing systematic.

---

## References


| Module | Location |
|--------|----------|
| Supervised Learning | sklearn/linear_model/, tree/, ensemble/, etc. |
| Unsupervised Learning | sklearn/cluster/, decomposition/, manifold/ |
| Data Transformation | sklearn/preprocessing/, feature_extraction/ |
| Model Selection | sklearn/model_selection/, metrics/ |
| Infrastructure | sklearn/utils/, externals/, tests/ |







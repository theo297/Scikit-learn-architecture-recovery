# Software Architecture Recovery: scikit-learn

**Course**: Software Architecture  
**Team**:
- Teophill Uwasan [2024326660270]
- Abhilasha Datta [2024326660084]
- Anastasia Gordienko [2024326660063]

**Date**: March 2026 - May 2026  
**Project Repository**: [Scikit-learn-architecture-recovery](https://github.com/theo297/Scikit-learn-architecture-recovery)

---

## Index

| # | Section |
|---|---------|
| 1 | [Project Overview](#1-project-overview) |
| 2 | [Architecture Recovery Method](#2-architecture-recovery-method) |
| 3 | [Module Structures](#3-module-structures) |
| 4 | [Component-and-Connector Structures](#4-component-and-connector-structures) |
| 5 | [Allocation Structures ](#5-allocation-structures) |
| 6 | [Early Architecture Decisions](#6-early-architecture-decisions) |
| 7 | [Quality Attributes](#7-quality-attributes) |
| 8 | [Business Context and Stakeholders](#8-business-context-and-stakeholders) |
| 9 | [Quality Attribute Tactics ](#9-quality-attribute-tactics) |
| 10 | [Patterns and Tactics ](#10-patterns-and-tactics) |
| 11 | [Activity Log](#11-activity-log) |
| 12 | [References](#12-references) |

---

## 1. Project Overview

scikit-learn is an open-source machine learning library for Python, built on NumPy, SciPy, and matplotlib. It provides tools for:

| Category | Examples |
|----------|----------|
| Supervised Learning | Classification, Regression |
| Unsupervised Learning | Clustering, Dimensionality Reduction |
| Model Selection | Cross-validation, Hyperparameter tuning |
| Preprocessing | Scaling, Normalization, Feature extraction |

**Key Facts:**

| Attribute | Details |
|-----------|---------|
| First release | 2007 (as scikits.learn) |
| Current version | 1.6.x |
| License | BSD 3-Clause |
| Core language | Python with Cython for performance-critical paths |
| Key dependencies | NumPy, SciPy, joblib |
| Primary maintainer | Inria (French National Institute for Research) |
| GitHub stars | 60,000+ |
| Monthly downloads | Millions |

**Why We Chose This Project:**
- Clean, consistent architecture built around a unified Estimator interface
- Well-documented design patterns including Strategy, Composite, and Template Method
- Ideal case study for software architecture recovery across all five course chapters

---

## 2. Architecture Recovery Method

| Method | Description |
|--------|-------------|
| Static code analysis | Analyzed scikit-learn source code version 1.6.x |
| Directory structure analysis | Mapped the sklearn/ directory hierarchy |
| Inheritance tracing | Traced BaseEstimator through all estimator subclasses |
| Cross-referencing | Verified every architectural claim against actual source code |

**Code Evidence Sources:**

| File | Lines | What It Contains |
|------|-------|------------------|
| sklearn/base.py | 67–260 | BaseEstimator: get_params, set_params |
| sklearn/base.py | 275–297 | ClassifierMixin, RegressorMixin, TransformerMixin |
| sklearn/pipeline.py | 434–700 | Pipeline class (Composite pattern) |
| sklearn/utils/validation.py | 925–1300 | check_array, check_X_y, check_is_fitted |
| sklearn/linear_model/_logistic.py | Full class | LogisticRegression implementation |
| sklearn/tree/_tree.pyx | Full file | Cython implementation of Decision Tree |

---

## 3. Module Structures 

Module structures describe how the system is decomposed into implementation units (modules), their responsibilities, and the relationships between them. Per Chapter 1, three module structure types are used: decomposition, layer, and class (generalization).

### 3.1 Decomposition Structure

The sklearn/ directory is decomposed into modules, each with a specific computational responsibility. This decomposition determines modifiability — changes to one module (e.g., adding a new solver to linear_model/) do not affect others.
```
sklearn/
├── base.py                    # BaseEstimator, Mixins — lines 67–297
├── linear_model/              # Supervised: linear algorithms
│   ├── _base.py               # Shared linear model base
│   ├── _logistic.py           # LogisticRegression
│   └── _ridge.py              # Ridge regression
├── ensemble/                  # Supervised: ensemble algorithms
│   ├── _forest.py             # RandomForestClassifier/Regressor
│   └── _gb.py                 # GradientBoostingClassifier
├── tree/                      # Supervised: tree-based algorithms
│   ├── _classes.py            # DecisionTreeClassifier
│   └── _tree.pyx              # Cython core (performance-critical)
├── svm/                       # Supervised: support vector machines
│   └── _classes.py            # SVC, SVR
├── cluster/                   # Unsupervised: clustering
│   └── _kmeans.py             # KMeans
├── decomposition/             # Unsupervised: dimensionality reduction
│   └── _pca.py                # PCA
├── preprocessing/             # Transformer: data scaling/encoding
│   └── _data.py               # StandardScaler, MinMaxScaler
├── feature_selection/         # Transformer: feature filtering
├── impute/                    # Transformer: missing value handling
├── pipeline.py                # Composition: Pipeline, FeatureUnion
├── utils/                     # Infrastructure: shared utilities
│   └── validation.py          # check_array, check_X_y
└── model_selection/           # Infrastructure: cross-validation
    └── _search.py             # GridSearchCV

```

**Module Classification:**

| Module Group | Type | Examples | Responsibility |
|---|---|---|---|
| linear_model/, ensemble/, tree/, svm/, neighbors/ | Supervised Learning | LogisticRegression, RandomForest, SVC | Learn from labeled data to predict |
| cluster/, decomposition/, manifold/, mixture/ | Unsupervised Learning | KMeans, PCA | Find patterns without labels |
| preprocessing/, feature_selection/, impute/ | Transformers | StandardScaler, SelectKBest | Transform data, no learning target |
| pipeline.py | Composition | Pipeline, FeatureUnion | Compose estimators into one object |
| utils/, model_selection/ | Infrastructure | check_array, GridSearchCV | Shared services used by all modules |
| base.py | Foundation | BaseEstimator, Mixins | Base contracts all modules implement |

### 3.2 Layer Structure

The layer structure shows which modules are allowed to use which other modules. Dependencies are strictly unidirectional — lower layers never depend on upper layers. This gives the system portability and modifiability.


DIA 1


**Dependency Rules (Layer Structure):**
- Algorithm modules (Supervised, Unsupervised, Transformer) depend on the Utility Layer for validation — never the reverse
- No circular dependencies exist between algorithm modules (linear_model/ does not depend on ensemble/)
- The Abstraction Layer (base.py) is the only layer all algorithm modules inherit from
- The Utility Layer has no dependency on any algorithm module — it is a pure service layer

### 3.3 Class Structure (Generalization / Inheritance Hierarchy)

The class structure shows inheritance (is-a) and composition (has-a) relationships between the major classes. This is the basis for scikit-learn's Strategy pattern: all estimators share a common interface through inheritance.


DIA 2


**Relationships Summary:**

| Relationship | Between | Type | Description |
|---|---|---|---|
| Inherits | All estimators → BaseEstimator | Generalization (is-a) | Gains get_params, set_params |
| Inherits | LogisticRegression → ClassifierMixin | Generalization (is-a) | Gains score() via accuracy |
| Inherits | StandardScaler → TransformerMixin | Generalization (is-a) | Gains fit_transform() |
| Composes | Pipeline → Estimators | Composition (has-a) | Holds list of (name, estimator) tuples |
| Uses | Algorithm modules → utils/ | Dependency | Calls check_array before processing |

### 3.4 Module Diagram — Structure and Relationships

This diagram combines decomposition, layer, and class views to show the complete module structure with explicitly labeled relationships. It is consistent with the text descriptions above.

<img width="724" height="1127" alt="Screenshot 2026-05-12 191109" src="https://github.com/user-attachments/assets/fe8108d2-ea79-4f56-b11f-7dfc6cbe0da5" />


---

## 4. Component-and-Connector Structures 

C&C structures describe how architectural elements interact at **runtime**. Per Chapter 1, C&C elements are components (runtime entities with state and behavior) connected by connectors (interaction mechanisms). The focus is on how the architecture enables communication — not on individual function calls.

### 4.1 Service Structure — Estimator as a Service Component

In scikit-learn, each estimator is a **service component** that accepts data through defined ports and returns results. The connector is the synchronous method-call protocol over NumPy array data. The Validation Utilities act as a gating connector — data cannot pass to the estimator without first traversing validation.


DIA 4


### 4.2 Concurrency Structure — Joblib Parallel Execution

When `n_jobs > 1` is set (e.g., in RandomForest or GridSearchCV), joblib.Parallel is the **connector** that distributes work across child processes. Components (estimator copies) run independently. This is the architecture of parallelism — not a function flow.


<img width="796" height="1154" alt="Screenshot 2026-05-12 191137" src="https://github.com/user-attachments/assets/048ba68e-9ad7-414a-b5f6-209ae1556f2b" />



### 4.3 Pipeline Composite Structure — Connector Chain

The Pipeline is a C&C structure where estimators are components and the data-passing protocol (NumPy array from one output port to the next input port) is the connector. This is a Pipe-and-Filter style at the component level.


<img width="828" height="444" alt="Screenshot 2026-05-12 191149" src="https://github.com/user-attachments/assets/057af6cd-26fa-4285-b0cc-c52475bf7b41" />


---

## 5. Allocation Structures — Chapter 1

Allocation structures map software elements to non-software environments (hardware, file systems, development teams).

### 5.1 Implementation Structure — GitHub Repository
scikit-learn/              ← Root repository
├── sklearn/               ← All source code (maps to module structure)
│   ├── base.py
│   ├── linear_model/
│   ├── ensemble/
│   └── ...
├── sklearn/tests/         ← Unit tests mirroring sklearn/ hierarchy
├── doc/                   ← Documentation source (Sphinx)
├── examples/              ← Runnable example scripts
├── benchmarks/            ← Performance benchmarking scripts
├── build_tools/           ← CI/CD scripts
└── CONTRIBUTING.md        ← Work assignment guidelines### 5.2 Deployment Structure

| Aspect | scikit-learn Configuration |
|--------|---------------------------|
| Execution platform | CPU only — no GPU by architectural decision |
| Installation | `pip install scikit-learn` or `conda install scikit-learn` |
| Runtime dependencies | NumPy, SciPy, joblib (auto-installed) |
| Operating systems | Windows, Linux, macOS (pure Python + compiled extensions) |
| External services | None — fully self-contained library |
| Packaging | Pre-built wheels with Cython extensions compiled per platform |

### 5.3 Work Assignment Structure (Conway's Law)

As described in Chapter 3, organizations that design systems produce architectures that mirror their communication structures. scikit-learn's module boundaries map directly to contributor team organization:

| Module Group | Assigned To | Communication Path |
|---|---|---|
| linear_model/, ensemble/, tree/ | Algorithm-specific maintainers | GitHub PR reviews per module |
| utils/ | Core maintainers only | All changes require core team approval |
| preprocessing/, impute/ | Preprocessing team | Separate issue labels |
| tests/ | All contributors | Required for every PR |
| doc/ | All contributors + doc team | Documentation PRs |

**Evidence:** CONTRIBUTING.md states that each module has designated module owners who must approve changes — a direct consequence of Conway's Law observed in the architecture.

---

## 6. Early Architecture Decisions 

Per Chapter 2, software architecture embodies the **earliest, hardest-to-change** design decisions. These decisions constrain all subsequent implementation. The high-level layer structure in Section 3.2 is the direct architectural manifestation of each decision below.

### 6.1 How the High-Level Architecture Expresses Early Decisions

The layer structure from Section 3.2 maps directly to these early decisions:

| Layer | Early Decision It Embodies |
|---|---|
| API Layer (fit/predict) | Decision 1: Unified Estimator Interface |
| Abstraction Layer (BaseEstimator) | Decision 2: BaseEstimator Inheritance |
| Algorithm Modules (tree/, svm/...) | Decision 3: Python + Cython |
| Deployment (no GPU tier) | Decision 4: CPU-only Execution |
| Utility Layer (NumPy arrays) | Decision 5: NumPy/SciPy as Core |

### Decision 1 — Unified Estimator Interface (fit / predict / transform)

| Aspect | Details |
|--------|---------|
| **What was decided** | Every estimator — regardless of algorithm — must implement fit() and predict() (or transform() for preprocessors) with a consistent signature |
| **Why this decision was made** | To allow users to learn one pattern and apply it to any algorithm; to enable meta-estimators (GridSearchCV, Pipeline) to work with any algorithm without knowing its internals |
| **Positive aspects (Pros)** | Users can replace LogisticRegression with RandomForestClassifier with a single line change · Automated tools (GridSearchCV) work with any estimator · Greatly reduces learning curve · Enables composability via Pipeline |
| **Negative aspects (Cons / Tradeoffs)** | Cannot expose algorithm-specific features through the public interface · Forces all algorithms into the same structural mold even when the fit/predict metaphor is unnatural · Some nuanced behaviors (e.g., online learning) must be wrapped artificially |
| **Why hard to change now** | Millions of users and thousands of downstream libraries depend on this contract · Changing it would break every existing script |
| **Where it appears in architecture** | The entire API Layer in Section 3.2; the interface of every class in Section 3.3 |
| **Code evidence** | `sklearn/base.py` — ClassifierMixin, RegressorMixin, TransformerMixin all enforce this contract |

### Decision 2 — BaseEstimator Inheritance with get_params / set_params

| Aspect | Details |
|--------|---------|
| **What was decided** | All estimators must inherit from BaseEstimator, which uses `inspect.signature(__init__)` to automatically introspect hyperparameters |
| **Why this decision was made** | Hyperparameter management must be automatic and consistent to enable GridSearchCV to enumerate and set parameters without any estimator-specific code |
| **Positive aspects (Pros)** | Automatic hyperparameter discovery — no manual registration needed · Enables GridSearchCV parameter search without estimator-specific code · Enables clone() for cross-validation · Serialization with version tracking warns on version mismatch |
| **Negative aspects (Cons / Tradeoffs)** | Forbids *args and **kwargs in __init__ — all parameters must be explicit · Forces a strict constructor contract that third-party libraries must follow · Adds inheritance coupling — estimators are tightly bound to BaseEstimator |
| **Why hard to change now** | Deeply embedded in lines 67–260 of base.py; every estimator in the codebase relies on it |
| **Where it appears in architecture** | The Abstraction Layer in Section 3.2; root of the class hierarchy in Section 3.3 |
| **Code evidence** | `sklearn/base.py` lines 67–260: `_get_param_names()` uses `inspect.signature` |

### Decision 3 — Python-first with Cython for Performance-Critical Paths

| Aspect | Details |
|--------|---------|
| **What was decided** | Write all algorithms in pure Python by default; use Cython (.pyx files) only for the innermost performance-critical loops |
| **Why this decision was made** | Python enables rapid development and a large contributor community; Cython bridges the gap for bottlenecks without requiring a full C++ rewrite |
| **Positive aspects (Pros)** | Python is readable, debuggable, and widely understood · Cython gives near-C performance for decision tree splitting, distance computations · Attracts a large open-source contributor community · Easy integration with the broader Python scientific ecosystem |
| **Negative aspects (Cons / Tradeoffs)** | Python interpreter overhead for non-critical paths · Cython code is harder to write, test, and debug · Two-language problem — contributors must know both Python and Cython · Pure C++ libraries (XGBoost, LightGBM) outperform on large datasets |
| **Why hard to change now** | Would require a complete rewrite of all Cython extensions and the surrounding Python wrappers |
| **Where it appears in architecture** | Algorithm modules layer (tree/_tree.pyx); Deployment structure (platform-specific compiled wheels) |
| **Code evidence** | `sklearn/tree/_tree.pyx` — full Cython implementation of the tree building algorithm |

### Decision 4 — CPU-only Execution Model

| Aspect | Details |
|--------|---------|
| **What was decided** | scikit-learn is designed exclusively for CPU execution; no GPU acceleration is provided |
| **Why this decision was made** | scikit-learn targets traditional (non-deep) machine learning where datasets fit in CPU memory; GPU programming would add deployment complexity without benefiting the target use cases |
| **Positive aspects (Pros)** | Works on any machine without GPU drivers or specialized hardware · No CUDA/ROCm dependency management · Simpler deployment and testing · Clear scope boundary from deep learning frameworks |
| **Negative aspects (Cons / Tradeoffs)** | Cannot leverage GPU for large-scale matrix operations · Slower for very large datasets compared to GPU-accelerated alternatives · Not competitive with GPU-based libraries for certain workloads (large neural nets, massive k-NN) |
| **Why hard to change now** | The entire data pipeline and memory model assumes NumPy CPU arrays; adding GPU support would require a fundamentally different memory and dispatch architecture |
| **Where it appears in architecture** | Deployment Structure (Section 5.2) — no GPU tier exists; Utility Layer uses CPU-based NumPy |
| **Code evidence** | No CUDA, ROCm, or GPU-specific code exists in the repository; partial Array API support added recently as an extension point |

### Decision 5 — NumPy/SciPy as Core Data Representation

| Aspect | Details |
|--------|---------|
| **What was decided** | All data exchanged between components must be NumPy arrays or SciPy sparse matrices; no other data formats are natively supported |
| **Why this decision was made** | NumPy and SciPy provide mature, optimized implementations of linear algebra, sparse matrices, and numerical operations that would be prohibitively expensive to reimplement |
| **Positive aspects (Pros)** | Access to highly optimized BLAS/LAPACK routines · Sparse matrix support for text and graph data · Seamless interoperability with pandas, matplotlib, and the broader Python scientific stack · Vectorized operations eliminate most Python-level loops |
| **Negative aspects (Cons / Tradeoffs)** | Large installation footprint (hundreds of MB) · Limits data representation to NumPy structures — other formats (Arrow, Torch tensors) require explicit conversion · Algorithms requiring non-array data structures are awkward to implement |
| **Why hard to change now** | check_array() and the entire validation layer assume NumPy arrays; changing this would require updating every estimator |
| **Where it appears in architecture** | Utility Layer (check_array enforces NumPy); connector protocol in C&C structures (Section 4) |
| **Code evidence** | `sklearn/utils/validation.py` line 925+: `check_array()` converts and validates input as NumPy arrays |

---

## 7. Quality Attributes 

Per Chapter 2, quality attributes are substantially determined by the architecture. The architecture either **enables** or **inhibits** each quality attribute.

### 7.1 Quality Attribute Analysis

| Quality Attribute | Architecture Status | How the Architecture Enables It | Concrete Evidence |
|---|---|---|---|
| **Usability** |  Fully Enabled | Unified fit/predict interface — one pattern for all algorithms | Switching from LogisticRegression to RandomForest requires changing only one class name |
| **Modifiability** |  Fully Enabled | Module decomposition isolates changes; BaseEstimator encapsulates parameter management | Adding a new estimator requires only implementing fit/predict — no changes to other modules |
| **Extensibility** | Fully Enabled | Clear inheritance contract allows third-party estimators | scikit-learn-contrib hosts 100+ community estimators using the same interface |
| **Performance** |  Partially Enabled | Cython for critical loops; joblib parallelism for independent tasks | `_tree.pyx` decision tree; `n_jobs` parameter in RandomForest, GridSearchCV |
| **Portability** |  Fully Enabled | Pure Python + platform-compiled wheels; no GPU or OS-specific dependencies | Runs identically on Windows, Linux, macOS |
| **Reusability** | High Value | Components can be reused independently; parts of components can be extracted and used separately | See Section 7.2 |
| **Testability** | Enabled | clone() creates independent copies; check_is_fitted() provides clear error states | cross_val_score clones estimator for each fold |
| **Security / Reliability** | Partially Enabled | Input validation via check_array gates all data entry; does not address adversarial inputs | `sklearn/utils/validation.py` |

### 7.2 Reusability — High Value

The architecture achieves high reusability at multiple levels:

**Level 1 — Full component reuse:** An entire module (e.g., preprocessing/) can be used independently of any learning algorithm. StandardScaler can be used purely for data transformation with no classifier involved.

**Level 2 — Partial component reuse (parts of components):** Even sub-parts of components can be extracted and reused. The clone() function (from base.py) can be applied to any estimator-like object, even custom ones. The check_array() validation utility is used independently by external libraries. BaseEstimator can be inherited by a completely new estimator outside of scikit-learn.

**Level 3 — Interface reuse:** The fit/predict protocol is so widely adopted that XGBoost, LightGBM, and Keras wrappers implement the same interface so they can be dropped into scikit-learn Pipelines.

| Component | Reusability Level | How It Is Reused |
|---|---|---|
| BaseEstimator | Full + Partial | Inherited by third-party estimators; clone() usable in isolation |
| StandardScaler | Full | Used standalone as a data preprocessor, independent of any model |
| Pipeline | Full | Used inside GridSearchCV, cross_val_score — any context accepting an estimator |
| check_array() | Partial (sub-component) | Imported directly by external libraries for their own validation |
| Metrics (accuracy_score, r2_score) | Full | Used in standalone evaluation scripts, not tied to any estimator |
| TransformerMixin.fit_transform() | Partial | Inherited selectively by any class implementing fit() + transform() |

**Evidence in code:**
- `preprocessing/` is imported and used without `linear_model/` in thousands of data pipelines
- `sklearn/base.py clone()` is called by external libraries independently
- scikit-learn-contrib hosts 100+ estimators that reuse only BaseEstimator and the mixin contracts

### 7.3 Quality Attribute Priority (from Business Context)

| Priority | Quality Attribute | Business Driver |
|---|---|---|
| 1 | Usability | Lowers barrier to entry; drives adoption across data scientists and students |
| 2 | Modifiability | Allows research community to add new algorithms rapidly |
| 3 | Reusability | Enables third-party ecosystem (scikit-learn-contrib, XGBoost wrappers) |
| 4 | Extensibility | Allows commercial organizations to build on top without forking |
| 5 | Performance | Competes with specialized libraries for production use cases |
| 6 | Portability | Works in any deployment environment |

---

## 8. Business Context and Stakeholders — Chapter 3

### 8.1 Business Context

scikit-learn exists to serve the scientific and data science communities as a free, high-quality, consistent machine learning library. It is sponsored by Inria and maintained by a global open-source community.

**Market Position:** scikit-learn differentiates from TensorFlow and PyTorch by focusing exclusively on traditional (non-deep) machine learning. It competes on API simplicity and consistency rather than raw performance.

**Business Model:** BSD 3-Clause license allows commercial use at no cost. Revenue comes indirectly through Inria funding and commercial sponsors rather than software sales.

**Target Users:**

| User Group | Primary Need | How Architecture Addresses It |
|---|---|---|
| Data Scientists | Productivity, reliability | Unified API reduces context switching between algorithms |
| Academic Researchers | Extensibility, reproducibility | BaseEstimator allows new algorithms to plug in cleanly |
| Students | Simplicity, learnability | One pattern (fit/predict) for all algorithms |
| Enterprises | Stability, maintainability | Strong testing culture, version warnings via __getstate__ |
| ML Engineers | Composability | Pipeline + GridSearchCV enable full ML workflow in one object |

### 8.2 Stakeholders and Concerns

| Stakeholder | Primary Concern | Specific Concern | How Architecture Addresses It |
|---|---|---|---|
| Data Scientists | Ease of use | Must switch algorithms quickly | Unified fit/predict interface |
| Researchers | Extensibility | Must add new algorithms | BaseEstimator inheritance contract |
| Enterprise Users | Reliability | Must not break existing pipelines | Strict API stability + InconsistentVersionWarning |
| Contributors | Clear extension points | Must know where to add code | Module decomposition by algorithm family |
| Inria (sponsor) | Research impact | Must be widely adopted | BSD license + usability |
| Students | Learning curve | Must understand one pattern | Consistent API across all 50+ estimators |
| Third-party library authors | Interoperability | Must integrate with scikit-learn | fit/predict contract is publicly documented |

### 8.3 Architecture Influence Cycle 

**Influences ON the Architecture (Context → Architecture):**

| Influence Source | Context Type | Specific Influence | Architectural Evidence |
|---|---|---|---|
| Academic ML research | Technical | New algorithms are published and must be added | HistGradientBoosting (2019), TargetEncoder (2023) added as new modules |
| User community feedback | Business | API pain points reported via GitHub issues | Metadata routing system added; set_output() API introduced |
| Python ecosystem growth | Technical | NumPy/SciPy ecosystem matured | Core data model committed to NumPy arrays |
| Hardware trends (GPU) | Technical | GPUs became dominant for deep learning | Explicit decision to remain CPU-only and not compete in that space |
| Open-source community | Project life-cycle | Contributors organize around module boundaries | Conway's Law — module ownership mirrors team structure |

**Influences FROM the Architecture (Architecture → Context):**

| Influence Target | Specific Influence | Evidence |
|---|---|---|
| Broader ML ecosystem | fit/predict protocol adopted industry-wide | XGBoost, LightGBM, Keras all implement sklearn-compatible interfaces |
| Contributor community | Teams self-organize around module boundaries | CONTRIBUTING.md module owner assignments |
| Commercial ecosystem | BSD license enabled enterprise adoption | Used in production at Google, Microsoft, and thousands of companies |
| Education | Taught in universities worldwide as the standard ML API | Included in every major ML textbook and MOOC |
| Third-party packages | scikit-learn-contrib ecosystem created | 100+ packages that extend sklearn without forking it |

---

## 9. Quality Attribute Tactics — Chapter 4

Per Chapter 4, an **architectural tactic** is a primitive design decision that directly affects a quality attribute response. Tactics are the atoms from which patterns are built. The seven categories of architectural design decisions from Chapter 4 are applied below.

### 9.1 Tactics Identified in scikit-learn

| Quality Attribute | Tactic | Where in Architecture | Why It Works |
|---|---|---|---|
| **Modifiability** | Encapsulation (Allocation of Responsibilities) | BaseEstimator hides parameter management behind get_params/set_params | Changes to parameter storage do not affect any estimator's algorithm code |
| **Modifiability** | Use an Intermediary (Coordination Model) | Pipeline sits between user code and individual estimators | Estimators can be swapped without the user changing any surrounding code |
| **Modifiability** | Restrict Communication Paths | Algorithm modules must not depend on each other; only on utils/ | A change in ensemble/ cannot break linear_model/ |
| **Performance** | Increase Available Resources (Resource Management) | joblib.Parallel with n_jobs distributes work across CPU cores | Independent subtasks (fold evaluations, tree building) run simultaneously |
| **Performance** | Reduce Computational Overhead (Resource Management) | Cython .pyx files replace Python for inner loops | Achieves near-C speed for decision tree node splitting and distance computation |
| **Availability** | Detect Faults (Data Model) | check_array(), check_X_y(), check_is_fitted() gate all data entry | Invalid inputs are caught at the architectural boundary before any computation |
| **Testability** | Record/Replay — Isolate Components | clone() creates a fresh, unfitted copy of any estimator | Each cross-validation fold receives an independent estimator — no state leakage |
| **Usability** | Maintain Semantic Coherence (Coordination Model) | All estimators expose identical interface regardless of internal algorithm | Users can read any scikit-learn example and apply the pattern immediately |
| **Extensibility** | Use Run-time Binding (Binding Time) | GridSearchCV receives an estimator object and binds parameters at runtime | New estimators work with GridSearchCV without any changes to GridSearchCV |
| **Portability** | Abstract Common Services (Choice of Technology) | TransformerMixin, ClassifierMixin provide default implementations | Third-party estimators gain score() and fit_transform() without re-implementing |

### 9.2 Seven Design Decision Categories Applied 

| Ch. 4 Category | scikit-learn Decision | Architectural Location |
|---|---|---|
| 1. Allocation of Responsibilities | BaseEstimator handles parameter management; algorithm classes handle only learning logic | base.py separation from algorithm modules |
| 2. Coordination Model | Synchronous call-return for training; asynchronous process-based parallel for n_jobs > 1 | Validation flow (sync); joblib (async multi-process) |
| 3. Data Model | All inter-component data exchanged as NumPy ndarray or scipy sparse matrix | check_array() at every module boundary |
| 4. Management of Resources | joblib process pool for parallelism; memory= parameter for caching in Pipeline | Pipeline.memory; joblib.Memory |
| 5. Mapping Among Architectural Elements | Each module maps to a team; each .pyx file maps to a compiled platform binary | Work assignment structure (Section 5.3) |
| 6. Binding Time | Estimator hyperparameters bound at fit() time (runtime); set_params() allows rebinding before fit | GridSearchCV binds C, kernel etc. before each fit call |
| 7. Choice of Technology | NumPy/SciPy for numerics; Cython for performance; joblib for parallelism; pytest for testing | All layers of the architecture |

---

## 10. Patterns and Tactics 

Per Chapter 13, patterns are packages of design decisions (collections of tactics) that solve recurring problems. Tactics are simpler than patterns. Patterns are underspecified with respect to real systems and must be augmented with additional tactics to address their weaknesses.

### 10.1 Patterns Identified in scikit-learn

| Pattern | Category | Context / Problem | Solution in scikit-learn | Code Location |
|---|---|---|---|---|
| **Strategy** | C&C / Module | Many ML algorithms solve the same problem differently; users need to swap them | All estimators implement the same fit/predict interface — algorithms are interchangeable strategies | Every estimator class |
| **Composite** | C&C | A sequence of preprocessing + learning steps must behave as a single estimator | Pipeline stores (name, estimator) pairs and delegates fit/predict through each step | sklearn/pipeline.py 434–700 |
| **Template Method** | Module | Every estimator must validate inputs before learning; validation logic should not be duplicated | BaseEstimator._fit_context() decorator runs _validate_params() before every fit() | sklearn/base.py |
| **Factory** | Module | Creating Pipelines with named steps is verbose; users need a shorthand | make_pipeline() auto-names steps from class names and returns a configured Pipeline | sklearn/pipeline.py |
| **Adapter** | Module | Existing Python functions (e.g., np.log) do not have the Transformer interface | FunctionTransformer wraps any callable into a Transformer with fit/transform | sklearn/preprocessing/_function_transformer.py |
| **Layer** | Module | Algorithm modules must not depend on each other; validation must be centralized | Strict layer structure prevents circular dependencies; utils/ serves all modules | Section 3.2 architecture |

### 10.2 Patterns as Packages of Tactics 

| Pattern | Tactics Bundled Inside | How They Combine |
|---|---|---|
| **Strategy** | Encapsulation + Maintain Semantic Coherence + Use Run-time Binding | Algorithm internals are encapsulated; the interface remains semantically identical; the specific algorithm is bound at instantiation time |
| **Composite (Pipeline)** | Use an Intermediary + Maintain Semantic Coherence + Restrict Communication Paths | Pipeline is the intermediary; it exposes the same interface as any single estimator; it prevents direct coupling between steps |
| **Template Method** | Encapsulation + Maintain Semantic Coherence | Validation skeleton is fixed (Template); algorithm-specific logic is encapsulated in subclass fit() |
| **Adapter (FunctionTransformer)** | Use an Intermediary + Validate Inputs | The adapter mediates between the raw function and the Transformer protocol; the validate parameter optionally runs check_array |
| **Layer** | Restrict Communication Paths + Encapsulation | No upward dependencies allowed; each layer encapsulates its own responsibility |

### 10.3 Tactics Augment Patterns — Addressing Weaknesses

| Pattern | Weakness | Augmenting Tactic | How scikit-learn Applies It |
|---|---|---|---|
| **Strategy** | Performance overhead — uniform interface may not allow algorithm-specific optimizations | Reduce Computational Overhead | Cython .pyx implementations bypass Python overhead for critical inner loops |
| **Strategy** | Cannot expose algorithm-specific features | Use Run-time Binding + Tags System | __sklearn_tags__() allows estimator-specific metadata without breaking the interface |
| **Composite (Pipeline)** | Adds indirection — each layer in the chain adds latency | Reduce Overhead | fit_transform() is overridden in many transformers to combine fit+transform in one pass |
| **Composite (Pipeline)** | Debugging is hard — errors inside a pipeline step are hard to trace | Detect Faults | check_is_fitted() gives clear error messages; Pipeline propagates step names in exceptions |
| **Factory (make_pipeline)** | May hide configuration — auto-naming obscures step names | Maintain Semantic Coherence | Named Pipeline constructor remains available; make_pipeline is a convenience, not required |
| **Adapter (FunctionTransformer)** | May lose type safety — wrapped functions have no type contracts | Validate Inputs | validate=True parameter causes check_array() to run on input before passing to the function |

### 10.4 Pattern Interactions — Architectural View


<img width="703" height="1366" alt="Screenshot 2026-05-12 191201" src="https://github.com/user-attachments/assets/228dbf99-8736-4664-a0f8-8e2cf0b65baf" />


---

## 11. Activity Log

| Week | Dates | Name | Work Done | Files Modified |
|------|-------|------|-----------|----------------|
| 1 | Mar 1–5, 2026 | Abhilasha Datta | Architecture recovery, system overview, design patterns initial analysis | architecture.md, system_overview.md, design_patterns.md |
| 2 | Mar 8–12, 2026 | Teophill Uwasan | BaseEstimator deep dive, validation system, pipeline architecture, module classification | base_estimator.md, validation.md, pipeline.md, modules.md |
| 3 | Mar 15–19, 2026 | Teophill Uwasan | Diagrams: module structure, quality attributes, stakeholders, influence cycle | module_structure.md, quality_attributes.md, stakeholders.md, influence_cycle.md |
| 4 | Mar 22–26, 2026 | Abhilasha Datta  | Integration of all documents, initial README creation, cross-referencing | README.md |
| 5 | Mar 29–Apr 2, 2026 | Abhilasha Datta | Chapter 3: Business context, stakeholders, architecture influence cycle | system_overview.md |
| 6 | Apr 5–9, 2026 | Teophill Uwasan | Corrected module classifications, fixed layer diagram, added relationship labels | modules.md, module_structure.md |
| 7 | Apr 12–16, 2026 | Teophill Uwasan | Chapter 4 tactics analysis, Chapter 13 patterns-and-tactics, pattern interactions diagram | README.md |
| 8 | Apr 19–23, 2026 | Abhilasha Datta  | Consolidated all documents into single README.md, resolved teacher feedback | README.md |
| 9 | Apr 26–30, 2026 | Abhilasha Datta  | Final review: added dates to log, corrected C&C diagrams to remove function flow, added reusability detail, fixed pattern interaction to architectural view | README.md |
| 10 | May 1–8, 2026 | Teophill Uwasan | Final submission preparation, index added, all teacher feedback incorporated | README.md |

---

## 12. References

| Resource | URL / Location |
|----------|---------------|
| scikit-learn GitHub repository | https://github.com/scikit-learn/scikit-learn |
| scikit-learn official documentation | https://scikit-learn.org/stable/ |
| BaseEstimator source | sklearn/base.py lines 67–260 |
| Mixin classes source | sklearn/base.py lines 275–297 |
| Pipeline source | sklearn/pipeline.py lines 434–700 |
| Validation source | sklearn/utils/validation.py lines 925–1300 |
| LogisticRegression source | sklearn/linear_model/_logistic.py |
| Decision Tree Cython source | sklearn/tree/_tree.pyx |
| Course textbook | Bass, Clements, Kazman — *Software Architecture in Practice*, 3rd Edition |
| CONTRIBUTING.md | https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md |

---

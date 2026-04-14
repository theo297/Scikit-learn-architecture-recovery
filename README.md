# Scikit-learn
Architecture recovery project for software architecture course


**Course**: Software Architecture  
**Team**: Uwasan Teophill [2024326660270], 
          Abhilasha Datta  [2024326660084], 
          Anastasia Gordienko [2024326660063].

## Project Overview
Recovering the architecture of scikit-learn, focusing on:
- Core Estimator system (BaseEstimator, Mixins)
- Module organization (linear_model, ensemble, preprocessing)
- Key design patterns
- Data validation flow

  
##  Document Index

| Document | Description | Link |
|----------|-------------|------|
| architecture.md | Main architecture recovery document | [Read](architecture.md) |
| system_overview.md | Executive summary + Chapter 3 | [Read](system_overview.md) |
| design_patterns.md | Design pattern analysis | [Read](design_patterns.md) |
| base_estimator.md | BaseEstimator deep dive | [Read](base_estimator.md) |
| validation.md | Validation system analysis | [Read](validation.md) |
| modules.md | Module classification | [Read](modules.md) |
| pipeline.md | Pipeline architecture | [Read](pipeline.md) |

## Diagrams
| Diagram | Description | Link|
|---------|---------|-------------|
|high_level_architecture| Layered architecture |[View](https://github.com/theo297/Scikit-learn-architecture-recovery/blob/56eeead848da6e1fda4b97a0e9400d32c9f0fff9/architecture.md#3-High-Level-Architecture-Diagram)|
|module_structure | Decomposition + layer + class | [View](---)|
|quality_attributes| Quality attribute analysis |[View](https://github.com/theo297/Scikit-learn-architecture-recovery/blob/0eb06606197e58b722b752338ba9703ffa0cd17b/quality_attributes.md)|
|stakeholders.png |  Diagram showing stakeholders (users, researchers, enterprises, contributors, sponsors) |[View](https://github.com/theo297/Scikit-learn-architecture-recovery/blob/0eb06606197e58b722b752338ba9703ffa0cd17b/stakeholders.md)|
|influence_cycle.png| Architecture influence cycle |[View](---)|
|pipeline_structure.png| Diagram showing data flow through pipeline transformers to final estimator | [View](https://github.com/theo297/Scikit-learn-architecture-recovery/blob/fb8c3536444a9f761f059d1f2ff1d2590b274cbb/pipeline.md#Data-Flow-Diagram) |
|component_relationships.png | Diagram showing inheritance and composition relationships between components | [View](https://github.com/theo297/Scikit-learn-architecture-recovery/blob/56eeead848da6e1fda4b97a0e9400d32c9f0fff9/architecture.md#7-Component-Relationships) |
|pattern_interactions_1.png | Diagram showing how Strategy, Composite, Template Method, Factory, Adapter interact | [View](https://github.com/theo297/Scikit-learn-architecture-recovery/blob/56eeead848da6e1fda4b97a0e9400d32c9f0fff9/design_patterns.md#8-Pattern-Interactions) |

---

## 3. Early Design Decisions

The following early architectural decisions shape scikit-learn's architecture:

| # | Decision | Benefits | Tradeoffs |
|---|----------|----------|-----------|
| 1 | Unified Estimator interface (fit/predict) | Easy to learn, interchangeable algorithms | Cannot expose algorithm-specific features easily |
| 2 | BaseEstimator inheritance with get_params/set_params | Enables GridSearchCV, consistent parameter management | Forces all estimators into same pattern |
| 3 | Python-first with Cython for performance | Easy to write, large ecosystem | Two-language problem, debugging difficulty |
| 4 | CPU-only execution model | Simple deployment, works everywhere | No GPU acceleration for large-scale tasks |
| 5 | NumPy/SciPy as core dependencies | Mature numerical libraries | Large dependencies, limited to NumPy structures |

**Subsystems identified:** 7 major subsystems (Estimator Core, Supervised Learning, Unsupervised Learning, Data Transformation, Model Selection, Composition, Infrastructure)

*For detailed analysis, see (architecture.md)*

---

## Design Patterns

The following design patterns are used in scikit-learn:

| Pattern | Location | Why Used |
|---------|----------|----------|
| **Strategy** | All estimators | Multiple algorithms for same task; runtime interchangeability |
| **Composite** | Pipeline | Compose multiple steps into single estimator; uniform treatment |
| **Template Method** | BaseEstimator | Consistent validation and fitting skeleton; code reuse |
| **Factory** | make_pipeline | Simplified pipeline creation; reduced boilerplate |
| **Adapter** | FunctionTransformer | Reuse existing functions as transformers; flexibility |

*For detailed analysis, see (design_patterns.md)*

---

## Three Structure Categories 

| Structure Type | Description | Evidence |
|----------------|-------------|----------|
| **Module Structures** | Decomposition, layer, class hierarchy | `sklearn/` directory, BaseEstimator inheritance |
| **C&C Structures** | Service (fit/predict), concurrency (joblib) | Pipeline, n_jobs parameter |
| **Allocation Structures** | Implementation (GitHub), deployment (CPU-only), work assignment (teams) | Repository structure, CONTRIBUTING.md |

*For detailed analysis, see (architecture.md)*

---

##  Quality Attributes 

| Priority | Quality Attribute | Status | How Achieved |
|----------|-------------------|--------|--------------|
| 1 | Usability |  Enabled | Unified fit/predict interface |
| 2 | Modifiability | Enabled | BaseEstimator isolates changes |
| 3 | Extensibility | Enabled | Clear extension points |
| 4 | Performance | Partial | Cython + joblib |
| 5 | Portability | Enabled | Pure Python + minimal dependencies |

*For detailed analysis, see (architecture.md)*

---

## Business Context and Stakeholders

**Key Stakeholders:**
- End Users (Data Scientists) → Ease of use
- Researchers → Extensibility
- Enterprise Users → Reliability
- Contributors → Clear extension points
- Sponsors (Inria) → Research impact

**Architecture Influence Cycle:**
- Contexts influence the architecture (technical, business, project life-cycle)
- The architecture influences back (shapes user expectations, team organization)

*For detailed analysis, see (system_overview.md)*

---

### Member 1: Uwasan Teophill [2024326660270]

**Role**: Core Infrastructure & Base System
- **Primary Focus**:
  - BaseEstimator class hierarchy and inheritance patterns
  - Mixin system (ClassifierMixin, RegressorMixin, TransformerMixin)
  - Cloning mechanism and parameter handling
  - Configuration system and global settings
- **Secondary Focus**: Pipeline architecture

### Member 2: Abhilasha Datta  [2024326660084]
**Role**: Data Flow & Validation
- **Primary Focus**:
  - Data validation flow (check_array, validate_data, check_X_y)
  - Input handling and preprocessing infrastructure
  - Utility functions in sklearn/utils/
  - Cross-validation mechanisms
- **Secondary Focus**: Testing infrastructure and estimator checks

### Member 3: Anastasia Gordienko [2024326660063]
**Role**: Module Organization & Algorithms
- **Primary Focus**:
  - Linear models module (LogisticRegression, linear_model/)
  - Ensemble methods (RandomForestClassifier, ensemble/)
  - Tree-based models and Cython implementations
  - Model selection (GridSearchCV, cross_validate)
- **Secondary Focus**: Meta-estimators (Pipeline, FeatureUnion)




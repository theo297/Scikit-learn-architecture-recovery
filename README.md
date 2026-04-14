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

  
## Repository Structure
scikit-learn-architecture-project/


├── README.md      

├── architecture.md  - Core architecture document   https://github.com/theo297/Scikit-learn-architecture-recovery/blob/56eeead848da6e1fda4b97a0e9400d32c9f0fff9/architecture.md

├── system_overview.md - Executive summary  https://github.com/theo297/Scikit-learn-architecture-recovery/blob/56eeead848da6e1fda4b97a0e9400d32c9f0fff9/design_patterns.md

├── design_patterns.md - Pattern analysis   https://github.com/theo297/Scikit-learn-architecture-recovery/blob/56eeead848da6e1fda4b97a0e9400d32c9f0fff9/system_overview.md

├── base_estimator.md - Deep dive (https://github.com/theo297/Scikit-learn-architecture-recovery/blob/fb8c3536444a9f761f059d1f2ff1d2590b274cbb/base_estimator.md)

├── validation.md - Validation system https://github.com/theo297/Scikit-learn-architecture-recovery/blob/fb8c3536444a9f761f059d1f2ff1d2590b274cbb/validation.md

├── modules.md - Module classification https://github.com/theo297/Scikit-learn-architecture-recovery/blob/fb8c3536444a9f761f059d1f2ff1d2590b274cbb/modules.md

│── pipeline.md - Pipeline architecture https://github.com/theo297/Scikit-learn-architecture-recovery/blob/fb8c3536444a9f761f059d1f2ff1d2590b274cbb/pipeline.md

├── high_level_architecture.png https://github.com/theo297/Scikit-learn-architecture-recovery/blob/56eeead848da6e1fda4b97a0e9400d32c9f0fff9/architecture.md#L93

├── module_structure.png

├── quality_attributes.png  https://github.com/theo297/Scikit-learn-architecture-recovery/blob/0eb06606197e58b722b752338ba9703ffa0cd17b/stakeholders.md

├── stakeholders.png https://github.com/theo297/Scikit-learn-architecture-recovery/blob/0eb06606197e58b722b752338ba9703ffa0cd17b/quality_attributes.md

├── influence_cycle.png

├── pipeline_structure.png  

├── component_relationships.png 

├── pattern_interactions_1.png


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




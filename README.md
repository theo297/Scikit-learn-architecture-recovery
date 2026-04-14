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
│
├── README.md                         
├── architecture.md  - Core architecture document  [arch.md](https://github.com/theo297/Scikit-learn-architecture-recovery/blob/7e0e0ea493dea6eb000671295e4523e3df7f8e00/arch.md)

├── system_overview.md - Executive summary     

├── design_patterns.md - Pattern analysis        

├── base_estimator.md - Deep dive

├── validation.md - Validation system
├── modules.md - Module classification
│── pipeline.md - Pipeline architecture
├── high_level_architecture.png
├── module_structure.png
├── quality_attributes.png
├── stakeholders.png
└── (other diagrams)


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




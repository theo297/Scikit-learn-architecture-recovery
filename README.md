# Scikit-learn
Architecture recovery project for software architecture course


**Course**: Software Architecture  
**Team**: Uwasan Teophill [2024326660270], 
          Abhilasha Datta  [2024326660084], 
          Anastasia Gordienko [2024326660063].
**Semester**: 2nd year, second semester 

## Project Overview
Recovering the architecture of scikit-learn, focusing on:
- Core Estimator system (BaseEstimator, Mixins)
- Module organization (linear_model, ensemble, preprocessing)
- Key design patterns
- Data validation flow

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

## Repository Structure
- `/docs` - Architecture documentation and findings
- `/diagrams` - Architecture diagrams (UML, flowcharts)
- `/notes` - Individual analysis notes

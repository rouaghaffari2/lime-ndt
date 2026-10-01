[![PyPI Downloads](https://static.pepy.tech/personalized-badge/lime-ndt?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/lime-ndt)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=aymen20002005_lime_ndt&metric=vulnerabilities)
![Bug](https://sonarcloud.io/api/project_badges/measure?project=aymen20002005_lime_ndt&metric=bugs)

# lime_ndt

lime_ndt is a Python library that introduces an enhanced version of the LIME technique for model explainability, leveraging Neural Decision Trees (NDTs) for improved local and global interpretability of machine learning models.

## Installation

```bash
pip install lime_ndt
```

## Quick Start (Classification)

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime_ndt import LimeNdtExplainer

# Load data and train a model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)

# Explain a prediction (NDT surrogate is used by default)
explainer = LimeNdtExplainer(X_train, feature_names=iris.feature_names,
                              class_names=iris.target_names.tolist(), mode='classification')
exp = explainer.explain_instance(X_test[0], rf.predict_proba)

# Visualize
exp.as_pyplot_figure()
```

## Regression Example

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from lime_ndt import LimeNdtExplainer

# Load data and train a model
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=42)
rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)

# Explain a prediction
explainer = LimeNdtExplainer(X_train, feature_names=diabetes.feature_names,
                              discretize_continuous=True, mode='regression')
exp = explainer.explain_instance(X_test[0], rf.predict, num_features=10)

# Show results
print(exp.as_list())
exp.as_pyplot_figure()
```

<img width="718" height="557" alt="image" src="https://github.com/user-attachments/assets/b59c98ec-2463-4a65-a465-5153d5137151" />

## Advanced Usage

You can customize the NDT surrogate parameters (gammas, tree depth, training epochs):

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime_ndt import LimeNdtExplainer, NDTRegressorWrapper

# Load data and train a model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)

# Create explainer
explainer = LimeNdtExplainer(X_train, feature_names=iris.feature_names,
                              class_names=iris.target_names.tolist(), mode='classification')

# Custom NDT surrogate with specific parameters
ndt = NDTRegressorWrapper(gammas=[1, 100], max_depth=10, epochs=50)
exp = explainer.explain_instance(X_test[0], rf.predict_proba, model_regressor=ndt)

print(exp.as_list())
```

## Features

- Enhanced LIME explanations using Neural Decision Trees.
- Support for tabular data (classification and regression).
- NDT surrogate used by default — no extra setup needed.
- Tools for extracting, analyzing, and comparing decision trees and forests.
- Utilities for discretization (quartile, decile, entropy).
- Integration with Keras and scikit-learn.

## Requirements

- Python 3.9+
- numpy, pandas, scikit-learn, keras, tensorflow, matplotlib

## How to cite

Mohamed Aymen Bouyahia and Argyris Kalogeratos, Enhancing LIME using Neural Decision Trees, arXiv preprint, http://arxiv.org/abs/2603.20919, 2026.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

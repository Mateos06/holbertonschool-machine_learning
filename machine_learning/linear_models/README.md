
<div align="center">

# LINEAR MODELS & KERNEL METHODS

### Statistical Learning · Regularization · Decision Boundaries · Explainable AI

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-7357D9?style=for-the-badge)

[Overview](#01--overview) ·
[Architecture](#02--learning-architecture) ·
[Algorithms](#03--mathematical-foundations) ·
[Experiments](#04--experimental-results) ·
[Explainability](#05--explainable-ai) ·
[Implementation](#06--implementation) ·
[Setup](#08--getting-started)

[← Machine Learning](../README.md) · [Main Repository](../../README.md)

---

<img src="assets/models.svg" width="95%" alt="Linear models and decision boundaries">

</div>

## 01 — Overview

Linear models represent one of the fundamental building blocks of machine learning.

This module investigates how statistical learning algorithms transform numerical features into continuous predictions and class probabilities, how regularization improves generalization, and how kernel methods overcome the limitations of linear decision boundaries.

The implementation consists of seven interconnected tasks using synthetic datasets designed to expose specific statistical and computational challenges.

<div align="center">

| Implementations | Model Configurations | Evaluation Metrics | Explainability Framework |
|:---:|:---:|:---:|:---:|
| **7** | **7** | **5** | **SHAP** |

</div>

### Research Questions

| Challenge | Approach |
|---|---|
| How can we estimate continuous relationships? | Ordinary Least Squares |
| How can we stabilize correlated coefficients? | Ridge Regularization |
| How can we identify relevant features? | Lasso Regularization |
| How can we measure prediction quality? | Regression and classification metrics |
| How can we interpret individual predictions? | SHAP |
| How can we estimate class membership? | Logistic Regression |
| How can we capture nonlinear decision boundaries? | SVM Kernel Methods |

> A model is not defined solely by its predictive performance, but also by its generalization, stability and interpretability.

---

## 02 — Learning Architecture

The tasks are organized into three interconnected components: regression, classification and model interpretation.

```mermaid
flowchart TD
    A["LINEAR MODELS & KERNEL METHODS"]

    A --> B["Regression"]
    A --> C["Classification"]

    B --> D["Task 0 — OLS"]
    B --> E["Task 2 — Ridge"]
    B --> F["Task 3 — Lasso"]

    D --> G["Task 1 — Evaluation"]
    E --> G
    F --> G

    D --> H["Task 4 — SHAP"]
    E --> H
    F --> H

    C --> I["Task 5 — Logistic Regression"]
    C --> J["Task 6 — SVM"]

    J --> K["Linear"]
    J --> L["Polynomial"]
    J --> M["RBF"]

    G --> N["Performance Analysis"]
    H --> O["Model Interpretation"]

    I --> P["Classification Comparison"]
    K --> P
    L --> P
    M --> P

    classDef root fill:#0F172A,stroke:#94A3B8,color:#FFFFFF
    classDef family fill:#1E3A8A,stroke:#60A5FA,color:#FFFFFF
    classDef model fill:#164E63,stroke:#22D3EE,color:#FFFFFF
    classDef evaluation fill:#14532D,stroke:#4ADE80,color:#FFFFFF
    classDef explain fill:#4C1D95,stroke:#A78BFA,color:#FFFFFF

    class A root
    class B,C family
    class D,E,F,I,J,K,L,M model
    class G,N,P evaluation
    class H,O explain
```

### Experimental Workflow

```mermaid
flowchart LR
    A["Synthetic Dataset"]
    B["Train / Test Split"]
    C["Initialize Model"]
    D["Fit Model"]
    E["Generate Predictions"]
    F["Evaluate"]
    G["Explain"]

    A --> B --> C --> D --> E --> F --> G

    classDef data fill:#1E293B,stroke:#94A3B8,color:#FFFFFF
    classDef model fill:#164E63,stroke:#22D3EE,color:#FFFFFF
    classDef result fill:#14532D,stroke:#4ADE80,color:#FFFFFF

    class A,B data
    class C,D,E model
    class F,G result
```

Each model constructor returns an unfitted estimator, allowing training, prediction and evaluation to be managed independently.

---

## 03 — Mathematical Foundations

### 3.1 Ordinary Least Squares

**Task 00 · Baseline Regression**

Ordinary Least Squares estimates a linear relationship between input features and a continuous target variable.

$$
\hat{y}=\beta_0+\sum_{j=1}^{p}\beta_jx_j
$$

The model minimizes the residual sum of squares:

$$
\hat{\beta}=
\arg\min_{\beta}
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

<details>
<summary><b>Explore the mathematical formulation</b></summary>

<br>

In matrix notation:

$$
y=X\beta+\epsilon
$$

When the design matrix has full column rank, the analytical solution is:

$$
\hat{\beta}=(X^TX)^{-1}X^Ty
$$

However, highly correlated features can make the matrix poorly conditioned, producing unstable coefficient estimates.

**Advantages**

- Simple mathematical formulation.
- Computationally efficient.
- Direct coefficient interpretation.
- Useful as a regression baseline.

**Limitations**

- Sensitive to influential outliers.
- Unstable under strong multicollinearity.
- Cannot directly capture nonlinear relationships without feature transformations.

</details>

```mermaid
flowchart LR
    A["Input Features"] --> B["Weighted Linear Sum"]
    B --> C["Prediction"]
    C --> D["Residual Error"]
    D --> E["Squared Error"]
    E --> F["Coefficient Estimation"]

    classDef process fill:#164E63,stroke:#22D3EE,color:#FFFFFF
    classDef result fill:#14532D,stroke:#4ADE80,color:#FFFFFF

    class A,B,C,D,E process
    class F result
```

[View Implementation →](0-linear_regression.py)

---

### 3.2 Ridge Regression

**Task 02 · L2 Regularization**

Ridge regression extends ordinary least squares by introducing a penalty on the squared magnitude of the coefficients.

$$
\hat{\beta}_{Ridge}
=
\arg\min_{\beta}
\left[
\|y-X\beta\|_2^2+
\alpha\|\beta\|_2^2
\right]
$$

The regularization parameter controls the balance between prediction error and coefficient magnitude.

```mermaid
flowchart LR
    A["Correlated Features"]
    A --> B["Ordinary Least Squares"]
    A --> C["Ridge Regression"]

    B --> D["Potentially Unstable Coefficients"]
    C --> E["L2 Penalty"]
    E --> F["Coefficient Shrinkage"]
    F --> G["Improved Stability"]

    classDef problem fill:#7C2D12,stroke:#FB923C,color:#FFFFFF
    classDef solution fill:#14532D,stroke:#4ADE80,color:#FFFFFF

    class B,D problem
    class C,E,F,G solution
```

<details>
<summary><b>Why does Ridge handle multicollinearity?</b></summary>

<br>

Multicollinearity occurs when two or more input variables contain strongly overlapping information.

Ordinary least squares may assign large and opposing coefficients to correlated features, producing unstable estimates.

Ridge reduces this sensitivity by penalizing large coefficient values.

Its matrix solution, assuming centered data and an unpenalized intercept, is:

$$
\hat{\beta}_{Ridge}
=
(X^TX+\alpha I)^{-1}X^Ty
$$

The additional diagonal term improves numerical conditioning when the regularization parameter is positive.

Ridge generally shrinks coefficients without setting them exactly to zero.

</details>

[View Implementation →](2-ridge_regression.py)

---

### 3.3 Lasso Regression

**Task 03 · L1 Regularization & Feature Selection**

Lasso introduces an absolute-value penalty that can force individual coefficients to become exactly zero.

$$
\hat{\beta}_{Lasso}
=
\arg\min_{\beta}
\left[
\frac{1}{2n}\|y-X\beta\|_2^2+
\alpha\|\beta\|_1
\right]
$$

This makes Lasso both a predictive algorithm and an embedded feature-selection method.

```mermaid
flowchart LR
    A["50 Input Features"]
    A --> B["Lasso Regression"]
    B --> C["L1 Regularization"]

    C --> D["6 Retained Features"]
    C --> E["44 Zero Coefficients"]

    D --> F["Sparse Predictive Model"]

    classDef input fill:#1E293B,stroke:#94A3B8,color:#FFFFFF
    classDef model fill:#164E63,stroke:#22D3EE,color:#FFFFFF
    classDef retained fill:#14532D,stroke:#4ADE80,color:#FFFFFF
    classDef removed fill:#7F1D1D,stroke:#F87171,color:#FFFFFF

    class A input
    class B,C model
    class D,F retained
    class E removed
```

<details>
<summary><b>How does Lasso perform feature selection?</b></summary>

<br>

The L1 penalty encourages sparsity in the coefficient vector.

Features whose coefficients become zero no longer contribute directly to the model's prediction.

For the provided synthetic experiment:

- Total input features: 50
- Informative features: 5
- Retained features: 6
- Zero coefficients: 44
- Feature reduction: 88%

Lasso does not guarantee that all retained features are genuinely informative. Its selection depends on the dataset, regularization strength and relationships between features.

</details>

[View Implementation →](3-Lasso_regression.py)

---

### Regularization Comparison

| Property | OLS | Ridge | Lasso |
|---|:---:|:---:|:---:|
| Penalty | None | L2 | L1 |
| Coefficient shrinkage | No | Yes | Yes |
| Sparse coefficients | No | Generally no | Yes |
| Embedded feature selection | No | No | Yes |
| Multicollinearity control | Limited | Strong | Possible |
| Regularization parameter | None | Alpha | Alpha |

```mermaid
flowchart TD
    A["Regression Problem"]
    A --> B{"Main Challenge?"}

    B -->|"Establish a baseline"| C["Ordinary Least Squares"]
    B -->|"Correlated predictors"| D["Ridge"]
    B -->|"Many irrelevant predictors"| E["Lasso"]

    C --> F["Unregularized Fit"]
    D --> G["Stable Coefficients"]
    E --> H["Sparse Coefficients"]

    classDef baseline fill:#1E293B,stroke:#94A3B8,color:#FFFFFF
    classDef ridge fill:#1E3A8A,stroke:#60A5FA,color:#FFFFFF
    classDef lasso fill:#14532D,stroke:#4ADE80,color:#FFFFFF

    class C,F baseline
    class D,G ridge
    class E,H lasso
```

*Conceptual model-selection guide. Final model selection should be based on validation performance.*

---

### 3.4 Logistic Regression

**Task 05 · Probabilistic Classification**

Despite its name, logistic regression is a classification algorithm.

It transforms a linear decision function into a probability using the sigmoid function.

$$
z=\beta_0+\beta^Tx
$$

$$
P(y=1|x)=\sigma(z)=\frac{1}{1+e^{-z}}
$$

```mermaid
flowchart LR
    A["Feature Vector"]
    A --> B["Linear Combination"]
    B --> C["Sigmoid Function"]
    C --> D["Class Probability"]
    D --> E{"Threshold ≥ 0.5?"}

    E -->|"Yes"| F["Class 1"]
    E -->|"No"| G["Class 0"]

    classDef model fill:#164E63,stroke:#22D3EE,color:#FFFFFF
    classDef positive fill:#14532D,stroke:#4ADE80,color:#FFFFFF
    classDef negative fill:#7F1D1D,stroke:#F87171,color:#FFFFFF

    class B,C,D model
    class F positive
    class G negative
```

<details>
<summary><b>Explore the sigmoid activation function</b></summary>

<br>

```mermaid
xychart-beta
    title "Sigmoid Function"
    x-axis [-6, -4, -2, -1, 0, 1, 2, 4, 6]
    y-axis "Probability" 0 --> 1
    line [0.0025, 0.0180, 0.1192, 0.2689, 0.5, 0.7311, 0.8808, 0.9820, 0.9975]
```

The sigmoid maps any real-valued number into a probability between zero and one.

For binary classification, a default threshold of 0.5 separates the two predicted classes.

The binary cross-entropy loss is:

$$
\mathcal{L}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right]
$$

The model learns coefficients that minimize its classification objective, including regularization when enabled.

</details>

[View Implementation →](5-logisitc_regression.py)

---

### 3.5 Support Vector Machines

**Task 06 · Maximum-Margin Classification**

Support Vector Machines construct decision boundaries by maximizing the margin between classes.

Kernel functions allow the model to represent nonlinear relationships without explicitly computing all transformed features.

```mermaid
flowchart TD
    A["Input Feature Space"]
    A --> B["Support Vector Machine"]

    B --> C["Linear Kernel"]
    B --> D["Polynomial Kernel"]
    B --> E["RBF Kernel"]

    C --> F["Linear Hyperplane"]
    D --> G["Polynomial Decision Boundary"]
    E --> H["Flexible Nonlinear Boundary"]

    F --> I["Class Prediction"]
    G --> I
    H --> I

    classDef root fill:#0F172A,stroke:#94A3B8,color:#FFFFFF
    classDef kernel fill:#312E81,stroke:#818CF8,color:#FFFFFF
    classDef result fill:#14532D,stroke:#4ADE80,color:#FFFFFF

    class A,B root
    class C,D,E kernel
    class I result
```

<details>
<summary><b>Linear Kernel — Mathematical Definition</b></summary>

<br>

$$
K(x_i,x_j)=x_i^Tx_j
$$

The linear kernel measures similarity through the dot product of two feature vectors.

It produces a linear decision boundary and is particularly useful when classes are approximately linearly separable.

</details>

<details>
<summary><b>Polynomial Kernel — Mathematical Definition</b></summary>

<br>

$$
K(x_i,x_j)=
(\gamma x_i^Tx_j+r)^d
$$

The polynomial kernel introduces higher-order feature interactions.

Its behavior depends on:

- Polynomial degree.
- Kernel coefficient gamma.
- Independent coefficient.
- Regularization parameter C.

</details>

<details>
<summary><b>Radial Basis Function — Mathematical Definition</b></summary>

<br>

$$
K(x_i,x_j)=
\exp(-\gamma\|x_i-x_j\|^2)
$$

The RBF kernel measures similarity based on the distance between observations.

Nearby observations have higher similarity, while distant observations have lower similarity.

This enables the model to construct highly flexible nonlinear decision boundaries.

</details>

| Kernel | Decision Boundary | Flexibility | Key Parameters |
|---|---|:---:|---|
| Linear | Linear hyperplane | Low | C |
| Polynomial | Polynomial boundary | Medium–High | C, degree, gamma, coef0 |
| RBF | Flexible nonlinear boundary | High | C, gamma |

[View Implementation →](6-svm.py)

---

## 04 — Experimental Results

The following results are obtained from the provided synthetic-data examples.

They demonstrate algorithm behavior under controlled conditions rather than universal performance rankings.

### 4.1 Baseline Regression

<div align="center">

| Model | RMSE | R² |
|:---:|:---:|:---:|
| Linear Regression | **4.95** | **0.827** |

</div>

The baseline establishes a reference point for evaluating the regularized regression models.

---

### 4.2 Ridge — Multicollinearity Experiment

**Scenario:** Two nearly identical features.

```mermaid
xychart-beta
    title "Ridge vs Ordinary Least Squares — R²"
    x-axis ["OLS", "Ridge"]
    y-axis "R² Score" 0 --> 1
    bar [0.827, 0.831]
```

| Model | R² |
|---|---:|
| Ordinary Least Squares | 0.827 |
| Ridge Regression | **0.831** |

**Observed difference:** +0.004 R², equivalent to 0.4 percentage points on a percentage-scaled R² score.

The results show a small predictive improvement with Ridge.

The primary theoretical advantage of Ridge in this experiment is coefficient stabilization rather than necessarily a large increase in predictive accuracy.

---

### 4.3 Lasso — Feature Selection Experiment

**Scenario:** 50 input features, of which only 5 are informative.

<div align="center">

| Baseline R² | Lasso R² | Feature Reduction |
|:---:|:---:|:---:|
| **0.489** | **0.819** | **88%** |

</div>

```mermaid
xychart-beta
    title "Impact of L1 Regularization — R²"
    x-axis ["OLS", "Lasso"]
    y-axis "R² Score" 0 --> 1
    bar [0.489, 0.819]
```

```mermaid
pie showData
    title Feature Selection with Lasso
    "Retained Features" : 6
    "Zero Coefficients" : 44
```

| Indicator | Result |
|---|---:|
| Original features | 50 |
| Informative features | 5 |
| Retained features | 6 |
| Zero coefficients | 44 |
| Feature reduction | 88% |
| Absolute R² improvement | +0.330 |

The regularized model achieves substantially better predictive performance while using considerably fewer features.

> This experiment illustrates how irrelevant features can negatively affect generalization and how regularization can help control model complexity.

---

### 4.4 Logistic Regression — Binary Classification

**Scenario:** Two informative features and two classes.

```mermaid
pie showData
    title Binary Classification Accuracy
    "Correct Predictions" : 88
    "Incorrect Predictions" : 12
```

<div align="center">

| Accuracy | Error Rate |
|:---:|:---:|
| **88%** | **12%** |

</div>

The model performs well in this particular binary classification setup, where a linear decision boundary is sufficiently effective.

---

### 4.5 SVM — Nonlinear Classification

**Scenario:** Three concentric classes.

Concentric class distributions expose the limitations of linear classification algorithms.

```mermaid
xychart-beta
    title "Classification Accuracy by Algorithm"
    x-axis ["Logistic", "Linear SVM", "Poly SVM", "RBF SVM"]
    y-axis "Accuracy (%)" 0 --> 100
    bar [27, 37, 45, 95]
```

| Algorithm | Accuracy | Error Rate |
|---|---:|---:|
| Logistic Regression | 27% | 73% |
| Linear SVM | 37% | 63% |
| Polynomial SVM | 45% | 55% |
| **RBF SVM** | **95%** | **5%** |

<div align="center">

| Best Model | Accuracy Improvement vs Logistic |
|:---:|:---:|
| **RBF SVM** | **+68 Percentage Points** |

</div>

```mermaid
flowchart TD
    A["Concentric Class Distribution"]
    A --> B["Linear Decision Boundary"]
    A --> C["Nonlinear Decision Boundary"]

    B --> D["Logistic Regression"]
    B --> E["Linear SVM"]

    C --> F["Polynomial SVM"]
    C --> G["RBF SVM"]

    D --> H["27% Accuracy"]
    E --> I["37% Accuracy"]
    F --> J["45% Accuracy"]
    G --> K["95% Accuracy"]

    classDef weak fill:#7F1D1D,stroke:#F87171,color:#FFFFFF
    classDef moderate fill:#78350F,stroke:#FBBF24,color:#FFFFFF
    classDef strong fill:#14532D,stroke:#4ADE80,color:#FFFFFF

    class H,I weak
    class J moderate
    class K strong
```

<details>
<summary><b>Why does the RBF kernel perform better?</b></summary>

<br>

The concentric classes cannot be effectively separated using straight decision boundaries.

Logistic regression and linear SVM are restricted to linear decision functions in the original feature space.

The RBF kernel introduces distance-based similarity, allowing the SVM to represent curved boundaries that better match the geometry of the data.

The experiment demonstrates that model selection should consider the underlying feature distribution, rather than relying only on algorithm popularity.

The reported results depend on the synthetic dataset and model configuration.

</details>

---

## 05 — Explainable AI

### SHAP — SHapley Additive exPlanations

**Task 04 · Model Interpretation**

SHAP explains a model prediction by assigning an individual contribution to every input feature relative to a baseline model output.

$$
f(x)=\phi_0+\sum_{j=1}^{p}\phi_j
$$

Where:

- $\phi_0$ represents the baseline model output.
- $\phi_j$ represents the contribution of feature $j$.
- $f(x)$ represents the explained model output.

### Explainability Architecture

```mermaid
flowchart TD
    A["Training Data"]
    B["Trained Model"]
    C["Test Observations"]

    A --> D["SHAP Explainer"]
    B --> D
    C --> D

    D --> E["SHAP Values"]

    E --> F["Local Explainability"]
    E --> G["Global Explainability"]

    F --> H["Individual Prediction"]
    G --> I["Feature Importance"]
    G --> J["Beeswarm Visualization"]

    classDef input fill:#1E293B,stroke:#94A3B8,color:#FFFFFF
    classDef explain fill:#4C1D95,stroke:#A78BFA,color:#FFFFFF
    classDef output fill:#14532D,stroke:#4ADE80,color:#FFFFFF

    class A,B,C input
    class D,E explain
    class F,G,H,I,J output
```

### Understanding a Prediction

The following values are illustrative and demonstrate how feature contributions combine with a baseline prediction.

```mermaid
flowchart LR
    A["Baseline: 50"]
    A --> B["Feature 1: +12"]
    B --> C["Feature 2: -5"]
    C --> D["Feature 3: +8"]
    D --> E["Prediction: 65"]

    classDef baseline fill:#1E293B,stroke:#94A3B8,color:#FFFFFF
    classDef positive fill:#14532D,stroke:#4ADE80,color:#FFFFFF
    classDef negative fill:#7F1D1D,stroke:#F87171,color:#FFFFFF
    classDef prediction fill:#312E81,stroke:#818CF8,color:#FFFFFF

    class A baseline
    class B,D positive
    class C negative
    class E prediction
```

<details>
<summary><b>Local vs. Global Explainability</b></summary>

<br>

| Local Explainability | Global Explainability |
|---|---|
| Explains one prediction | Explains overall model behavior |
| Individual feature contributions | Aggregated feature contributions |
| Investigates unusual predictions | Identifies systematic patterns |
| Waterfall or force plots | Beeswarm and importance plots |

SHAP describes how a trained model uses its features. It does not establish causal relationships between variables.

</details>

<details>
<summary><b>Why use training data as the background?</b></summary>

<br>

The background dataset establishes a reference distribution against which predictions are explained.

Using training data avoids defining the reference directly from the test observations being evaluated.

The implemented function:

1. Receives an already trained model.
2. Creates an explainer using training data.
3. Computes SHAP values for test observations.
4. Returns the explainer and computed SHAP values.

</details>

[View Implementation →](4-shap.py)

---

## 06 — Implementation

### Task Directory

| ID | Implementation | Objective |
|:---:|---|---|
| 00 | [Linear Regression](0-linear_regression.py) | Baseline continuous prediction |
| 01 | [Evaluation Metrics](1-regression_evaluation_metrics.py) | Regression performance measurement |
| 02 | [Ridge Regression](2-ridge_regression.py) | L2 regularization |
| 03 | [Lasso Regression](3-Lasso_regression.py) | L1 regularization and sparsity |
| 04 | [SHAP](4-shap.py) | Prediction interpretation |
| 05 | [Logistic Regression](5-logisitc_regression.py) | Probabilistic classification |
| 06 | [Support Vector Machines](6-svm.py) | Linear and nonlinear classification |

### Function Interface

<details>
<summary><b>Task 00 — Linear_Regression()</b></summary>

<br>

```python
Linear_Regression()
```

**Returns:** An unfitted `sklearn.linear_model.LinearRegression` estimator.

**Purpose:** Establish a baseline regression model.

</details>

<details>
<summary><b>Task 01 — evaluation_metrics_for_regression()</b></summary>

<br>

```python
evaluation_metrics_for_regression(y_true, y_pred)
```

**Parameters**

- `y_true`: Actual target values.
- `y_pred`: Predicted target values.

**Returns**

```python
(mse, rmse, mae, r2)
```

</details>

<details>
<summary><b>Task 02 — ridge_regression()</b></summary>

<br>

```python
ridge_regression(random_state)
```

**Returns:** An unfitted `Ridge` estimator.

**Purpose:** Stabilize regression coefficients using L2 regularization.

</details>

<details>
<summary><b>Task 03 — lasso_regression()</b></summary>

<br>

```python
lasso_regression(random_state)
```

**Returns:** An unfitted `Lasso` estimator.

**Purpose:** Apply L1 regularization and encourage sparse coefficients.

</details>

<details>
<summary><b>Task 04 — get_shap_explainer_and_values()</b></summary>

<br>

```python
get_shap_explainer_and_values(
    model,
    X_train,
    X_test
)
```

**Parameters**

- `model`: A trained model.
- `X_train`: Background training data.
- `X_test`: Observations to explain.

**Returns:** SHAP explainer and computed SHAP values.

</details>

<details>
<summary><b>Task 05 — Logistic_Regression_Model()</b></summary>

<br>

```python
Logistic_Regression_Model(random_state)
```

**Returns:** An unfitted `LogisticRegression` estimator.

**Purpose:** Estimate class probabilities and predict class membership.

</details>

<details>
<summary><b>Task 06 — get_SVM_model()</b></summary>

<br>

```python
get_SVM_model(name, random_state)
```

**Supported kernels**

```python
"linear"
"poly"
"rbf"
```

**Returns:** An unfitted `SVC` estimator.

**Exceptions:** Raises `ValueError` for unsupported kernel names.

</details>

---

### Task Dependencies

```mermaid
flowchart LR
    T0["00 · Linear Regression"]
    T1["01 · Metrics"]
    T2["02 · Ridge"]
    T3["03 · Lasso"]
    T4["04 · SHAP"]
    T5["05 · Logistic"]
    T6["06 · SVM"]

    T0 --> T1
    T2 --> T1
    T3 --> T1

    T0 --> T4
    T2 --> T4
    T3 --> T4

    T5 --> C["Classifier Comparison"]
    T6 --> C

    classDef regression fill:#164E63,stroke:#22D3EE,color:#FFFFFF
    classDef evaluation fill:#14532D,stroke:#4ADE80,color:#FFFFFF
    classDef classification fill:#312E81,stroke:#818CF8,color:#FFFFFF

    class T0,T2,T3 regression
    class T1,T4,C evaluation
    class T5,T6 classification
```

*Arrows indicate conceptual and experimental relationships, not necessarily direct Python imports.*

---

## 07 — Evaluation Framework

### Regression Metrics

```mermaid
flowchart TD
    A["Actual Values"]
    B["Predicted Values"]

    A --> C["Prediction Error"]
    B --> C

    C --> D["MAE"]
    C --> E["MSE"]
    C --> F["RMSE"]
    C --> G["R²"]

    D --> H["Absolute Error"]
    E --> I["Squared Error"]
    F --> J["Original Target Units"]
    G --> K["Relative Model Fit"]
```

| Metric | Mathematical Meaning | Objective |
|---|---|:---:|
| MAE | Mean absolute error | Minimize |
| MSE | Mean squared error | Minimize |
| RMSE | Root mean squared error | Minimize |
| R² | Fit relative to a mean baseline | Maximize |
| Accuracy | Fraction of correct classifications | Maximize |

<details>
<summary><b>View mathematical definitions</b></summary>

<br>

**Mean Absolute Error**

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

**Mean Squared Error**

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

**Root Mean Squared Error**

$$
RMSE=\sqrt{MSE}
$$

**Coefficient of Determination**

$$
R^2=1-
\frac{\sum_i(y_i-\hat{y}_i)^2}
{\sum_i(y_i-\bar{y})^2}
$$

**Classification Accuracy**

$$
Accuracy=\frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

An R² value of 1 indicates perfect predictions. A negative value indicates performance worse than the mean-prediction reference.

</details>

[View Metrics Implementation →](1-regression_evaluation_metrics.py)

---

## 08 — Getting Started

### Directory Structure

```text
linear_models/
│
├── assets/
│   └── models.svg
│
├── 0-linear_regression.py
├── 1-regression_evaluation_metrics.py
├── 2-ridge_regression.py
├── 3-Lasso_regression.py
├── 4-shap.py
├── 5-logisitc_regression.py
├── 6-svm.py
│
└── README.md
```

### Environment Setup

Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate the environment.

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install numpy scikit-learn matplotlib shap
```

### Running Experiments

The `N-main.py` scripts are provided with the original project and are not included in this repository.

Place the corresponding main script inside this directory and execute it:

```bash
python 3-main.py
```

### Example Usage

```python
from sklearn.model_selection import train_test_split

Linear_Regression = __import__(
    "0-linear_regression"
).Linear_Regression

evaluate = __import__(
    "1-regression_evaluation_metrics"
).evaluation_metrics_for_regression

# X and y represent the dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Linear_Regression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse, rmse, mae, r2 = evaluate(
    y_test,
    predictions
)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
```

> Run the examples from the `linear_models/` directory because the task modules use numbered filenames and dynamic imports.

### Implementation Conventions

- Model constructors return unfitted estimators.
- Training is performed externally using `.fit()`.
- Predictions are generated using `.predict()`.
- `random_state` is passed through where required.
- Unsupported SVM kernels raise `ValueError`.
- Feature scaling is recommended for regularized models and SVMs.
- Preprocessing should be fitted on training data only.

The filename `5-logisitc_regression.py` intentionally preserves the spelling required by the original main script.

---

## 09 — Key Findings

```mermaid
flowchart TD
    A["Experimental Findings"]

    A --> B["Regularization"]
    A --> C["Feature Selection"]
    A --> D["Decision Boundaries"]
    A --> E["Interpretability"]

    B --> B1["Ridge improves coefficient stability"]
    C --> C1["Lasso reduces irrelevant features"]
    D --> D1["RBF captures nonlinear structure"]
    E --> E1["SHAP explains feature contributions"]

    classDef root fill:#0F172A,stroke:#94A3B8,color:#FFFFFF
    classDef concept fill:#164E63,stroke:#22D3EE,color:#FFFFFF
    classDef conclusion fill:#14532D,stroke:#4ADE80,color:#FFFFFF

    class A root
    class B,C,D,E concept
    class B1,C1,D1,E1 conclusion
```

| Experiment | Main Observation |
|---|---|
| Ridge | Small R² improvement in the correlated-feature experiment |
| Lasso | Higher R² with 88% fewer nonzero feature coefficients |
| Logistic Regression | 88% accuracy on the binary classification example |
| RBF SVM | 95% accuracy on the concentric-class example |
| SHAP | Feature-level interpretation of model predictions |

These experiments demonstrate that the most suitable model depends on the structure of the data, the underlying prediction objective and the required level of interpretability.

---

## 10 — Competency Map

| Technical Domain | Skills Developed |
|---|---|
| Statistical Learning | Least squares, coefficient estimation |
| Optimization | Loss functions and regularization |
| Feature Engineering | Multicollinearity and sparse feature selection |
| Classification | Probability estimation and decision boundaries |
| Kernel Methods | Linear, polynomial and radial basis functions |
| Model Evaluation | MSE, RMSE, MAE, R² and accuracy |
| Explainable AI | SHAP values and feature contributions |
| Software Engineering | Modular estimators, function interfaces and reproducibility |

---

<div align="center">

## Continue Exploring

<table>
<tr>
<td align="center" width="50%">

<b>Machine Learning</b>

<br>

Explore the complete machine learning architecture.

<br><br>

<a href="../README.md">Return to Overview →</a>

</td>
<td align="center" width="50%">

<b>Tree-Based Models</b>

<br>

Explore recursive partitioning, pruning and ensemble learning.

<br><br>

<a href="../tree_models/README.md">Explore Tree Models →</a>

</td>
</tr>
</table>

---

**MATHEMATICS → IMPLEMENTATION → EXPERIMENTATION → INTERPRETATION**

*Understanding the model beyond its predictions.*

[Repository](../../README.md) · [Machine Learning](../README.md) · [Tree Models](../tree_models/README.md)

</div>

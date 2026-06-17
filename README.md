# Social Network Ads — Purchase Predictor

A machine learning project that predicts whether a social media user **will purchase a product** after viewing an ad, based on **age** and **estimated salary**. The script compares four classifiers, evaluates them with cross-validation, and generates decision-boundary visualizations.

---

## Overview

This is an improved version of a basic logistic-regression-only approach. It adds:

- Feature scaling with `StandardScaler`
- Side-by-side comparison of **four classification algorithms**
- 5-fold cross-validation for robust accuracy estimates
- Confusion matrix and classification report for the best model
- Decision boundary and distribution plots
- A sample prediction for a new user profile

---

## Dataset

| Property | Value |
|----------|-------|
| **File** | `Social_Network_Ads.csv` |
| **Samples** | ~400 |
| **Features used** | `Age`, `EstimatedSalary` |
| **Target** | `Purchased` — 0 = no, 1 = yes |

Columns `User ID` and `Gender` are present in the CSV but not used as model inputs.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualization | matplotlib |

---

## Project Structure

```
purchase pridector/
├── social_ads_prediction.py      # Training, comparison, and visualization pipeline
├── Social_Network_Ads.csv          # Dataset
└── README.md
```

**Generated at runtime:**

- `model_comparison.png` — cross-validation accuracy bar chart
- `decision_boundary.png` — SVM decision regions (train and test)
- `distribution.png` — age and salary histograms by purchase outcome

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Run

```bash
cd "purchase pridector"
python social_ads_prediction.py
```

---

## Models Compared

| Model | Preprocessing | Key Parameters |
|-------|---------------|----------------|
| Logistic Regression | Scaled | Default |
| Decision Tree | Raw features | `max_depth=5`, `random_state=42` |
| Random Forest | Raw features | `n_estimators=100`, `random_state=42` |
| SVM (RBF) | Scaled | `kernel='rbf'`, `random_state=42` |

The pipeline uses an 80/20 train/test split (`random_state=42`) and reports both test accuracy and 5-fold cross-validation scores.

**Best performer:** SVM (RBF) — typically achieves ~92.5% accuracy on this dataset.

---

## Pipeline

```text
Load Social_Network_Ads.csv
    │
    ▼
Train/test split (80/20)
    │
    ├─► Scale features (LR + SVM)
    │
    ├─► Train 4 classifiers + cross-validation
    │
    ├─► Deep-dive on SVM (confusion matrix, classification report)
    │
    ├─► Save visualizations (comparison, boundaries, distributions)
    │
    └─► Predict for sample user (Age=30, Salary=87,000)
```

---

## Features

- **Multi-model benchmark** — compare linear, tree-based, ensemble, and kernel methods on the same split
- **Scaling awareness** — demonstrates when `StandardScaler` matters (LR, SVM) vs when it does not (trees)
- **Visual decision boundaries** — 2D meshgrid plots for intuitive model comparison
- **Inference demo** — end-to-end prediction for a hypothetical new ad viewer

---

## Example Console Output

```
SOCIAL NETWORK ADS — PURCHASE PREDICTION
Dataset shape : (400, 5)
Purchased=1   : 143 people
Purchased=0   : 257 people

Model comparison table (accuracy + CV scores)
SVM confusion matrix and classification report
New user prediction: Age=30, Salary=87000 → Purchased: Yes/No
```

---

## Limitations

- Only two features used (gender and user ID ignored)
- Models are not saved to disk
- No REST API or deployment layer
- Offline batch script only

---

## License

Educational and portfolio use. The Social Network Ads dataset is widely used in ML tutorials and courses.

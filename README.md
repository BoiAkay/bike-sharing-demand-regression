# 🚲 Bike Sharing Demand Prediction

**BITS Pilani WILP — M.Tech (AI/ML) | Machine Learning Assignment 1**

> Predicting hourly bike rental demand using weather, time, and seasonal data with Linear Regression and its extensions.

---

## 📋 Assignment Overview

| Field | Details |
|---|---|
| **Course** | Machine Learning — BITS Pilani WILP (M.Tech AI/ML) |
| **Student** | Ayush Kumar Gupta |
| **Student ID** | 2025AA05064 |
| **Evaluation Metric** | RMSLE (Root Mean Squared Logarithmic Error) |
| **Best Score** | **0.2619 RMSLE** (Decision Tree Regressor, depth=6) |

---

## 🗂️ Repository Structure

```
.
├── final_ml_VLab_assignment.ipynb   # Main notebook with all analysis & models
├── assignment_dataset/
│   ├── bike_train.csv               # Training data (10,450 rows × 12 features)
│   ├── bike_test.csv                # Test data (no target column)
│   └── sampleSubmission.csv        # Expected submission format
├── assignment_images/
│   ├── 01_correlation_matrix.png
│   ├── 02_categorical_boxplots.png
│   ├── 03_hourly_trends.png
│   ├── 04_continuous_features.png
│   └── decision-tree-residual_plot.png
├── assignment_output/
│   └── submission.csv              # Final predictions on test set
└── README.md
```

---

## 📊 Dataset

The dataset comes from the **Bike Sharing Demand** challenge and contains hourly rental records with the following features:

| Feature | Type | Description |
|---|---|---|
| `datetime` | Object (Timestamp) | Date and time of the record |
| `season` | Integer (Categorical) | 1=Spring, 2=Summer, 3=Fall, 4=Winter |
| `holiday` | Integer (Categorical) | Whether the day is a holiday |
| `workingday` | Integer (Categorical) | Whether the day is a working day |
| `weather` | Integer (Categorical) | Weather condition (1=Clear → 4=Heavy Rain) |
| `temp` | Float | Actual temperature (°C) |
| `atemp` | Float | "Feels like" temperature (°C) |
| `humidity` | Integer | Relative humidity |
| `windspeed` | Float | Wind speed |
| `casual` | Integer | Count of casual (non-registered) users |
| `registered` | Integer | Count of registered users |
| `count` | Integer | **Target** — total hourly rentals |

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights from the analysis:

- **No missing values** in the dataset.
- **`hour` (+0.40)** and **`temp` (+0.40)** are the strongest predictors of demand.
- **`humidity` (-0.32)** has a notable negative correlation with rentals.
- **`atemp` and `temp`** are nearly identical (correlation: 0.98) — `atemp` was dropped to avoid multicollinearity.
- **`season` and `month`** are highly redundant (correlation: 0.97) — `season` was dropped.
- `workingday` and `holiday` show near-zero linear correlation but are critical for capturing hourly *shape* of demand (commute spikes vs. leisure).

---

## ⚙️ Feature Engineering

Four techniques were applied to improve model performance:

1. **Cyclical Encoding** — `hour` and `month` encoded as sine/cosine pairs to preserve the circular nature of time.
2. **One-Hot Encoding** — `weather` converted to dummy variables; test set columns aligned to training set to prevent data leakage.
3. **Polynomial Features (degree=2)** — Interaction and squared terms for `temp`, `humidity`, and `windspeed`.
4. **Log Transformation of Target** — `np.log1p(count)` applied to reduce skewness and stabilize variance.

---

## 🤖 Models & Results

All models were evaluated using RMSLE on a validation split (70/30).

| Model | RMSLE | Key Observations |
|---|:---:|---|
| Linear Regression | 0.8103 | Underfits; cannot capture non-linear demand patterns |
| Poly Regression + Ridge (L2) | 0.4353 | Significant improvement; regularization handles multicollinearity from poly features |
| Poly Regression + Lasso (L1) | 0.4401 | Comparable to Ridge; slight coefficient sparsity but marginally worse |
| Hist Gradient Boosting | 0.3647 | Strong performance; captures complex interactions automatically |
| **Decision Tree (depth=6)** | **0.2619** | **Best model**; hierarchical splits handle discontinuities in demand perfectly |

### 🏆 Why Decision Tree Won

The Decision Tree Regressor (max_depth=6) achieved the lowest RMSLE by:
- **Handling discontinuities**: Real demand has sharp "steps" (e.g., sudden rush-hour spikes) that polynomial curves cannot fit cleanly.
- **Hierarchical segmentation**: Naturally isolates combinations like *"Is it raining? → Is it rush hour?"* without explicit feature engineering.
- **Depth constraint**: A depth of 6 provides enough complexity to capture demand patterns while limiting overfitting.

---

## 📐 Evaluation Metric

```python
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)  # clip negatives
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))
```

RMSLE was chosen over RMSE because:
- It focuses on **relative error** rather than absolute scale, making it robust to large-count outliers.
- It compresses the prediction scale via logarithms, preventing rare high-demand hours from dominating the loss.

---

## 🛠️ Tech Stack

- **Python 3.x**
- `pandas`, `numpy` — Data manipulation
- `scikit-learn` — Models, pipelines, GridSearchCV
- `matplotlib`, `seaborn` — Visualization
- `joblib` — Model persistence

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn joblib
   ```

3. **Download the datasets** from the links below and place them in `assignment_dataset/`:
   - [bike_train.csv](https://drive.google.com/file/d/18kpPgnoDfOfx3PRRZY3xbsLeWalNViYl/view)
   - [bike_test.csv](https://drive.google.com/file/d/1HJlgPTkGlCzycSTnJNB0oJEfgjSV4o0f/view)
   - [sampleSubmission.csv](https://drive.google.com/file/d/1g_hQQtI80Dz0NkGXG8ZioND9iVGTerd8/view)

4. **Run the notebook**
   ```bash
   jupyter notebook final_ml_VLab_assignment.ipynb
   ```

---

## 📝 Reflection Highlights

- **Why RMSLE over RMSE?** RMSLE penalizes large errors on a logarithmic scale, making it resilient to outliers and focusing on proportional accuracy rather than raw magnitude.
- **Simplicity vs. Power trade-off**: Simple models (Linear Regression) are interpretable but underfit; complex models (trees, boosting) achieve higher accuracy at the cost of explainability and risk of overfitting.
- **Why Linear Regression fails for time-of-day?** It assumes a monotonic relationship between hour and count, missing the bimodal rush-hour pattern and the cyclical nature of time entirely.

---

*Submitted for BITS Pilani WILP M.Tech (AI/ML) Machine Learning — Assignment 1*

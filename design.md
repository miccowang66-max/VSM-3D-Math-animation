# Project Design Document — Multiple Linear Regression on `50_Startups`

> **Project:** Profit Prediction via Multiple Linear Regression  
> **Dataset:** `50_Startups.csv` (R&D Spend, Administration, Marketing Spend, State, Profit)  
> **Goal:** Build, train, and evaluate a Multiple Linear Regression model to predict startup profit.

---

## 1. Technology Stack

| Layer            | Technology / Library    |
| ---------------- | ----------------------- |
| Language         | Python 3.10+            |
| Data Processing  | Pandas, NumPy           |
| ML Framework     | scikit-learn            |
| Visualization    | Matplotlib, Seaborn     |
| Statistics       | statsmodels (optional)  |
| Environment      | venv / conda            |
| Version Control  | Git                     |

---

## 2. Directory Tree

```
project_root/
│
├── design.md                      # ← This file (single source of truth)
├── README.md                      # Project overview and run instructions
├── requirements.txt               # Pinned dependencies
│
├── data/
│   ├── raw/                       # 📦 READ-ONLY — Original, immutable dataset
│   │   └── 50_Startups.csv
│   │
│   └── processed/                 # 📝 Transformed datasets written by pipeline
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   └── 01_eda.ipynb               # 📖 READ-ONLY DISPLAY — Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py               # 🧹 CONSOLIDATED — All preprocessing logic
│   ├── train.py                   # 🏋️ Model training
│   ├── evaluate.py                # 📊 Model evaluation & metrics
│   └── utils.py                   # 🔧 Shared helper functions
│
├── models/
│   └── linear_regression.pkl      # Serialized trained model
│
├── results/
│   ├── metrics.json               # Evaluation metrics (R², RMSE, MAE, etc.)
│   └── plots/                     # Generated charts & visualizations
│       ├── residuals.png
│       └── actual_vs_predicted.png
│
└── tests/
    ├── test_data_prep.py
    └── test_train.py
```

---

## 3. Pipeline Stages (Sequential)

### Stage 0 — Data Ingestion & Display *(READ-ONLY)*

- **Owner:** `notebooks/01_eda.ipynb`
- **Dataset:** `data/raw/50_Startups.csv`
- **Rules (STRICT):**
  - The raw CSV file is **immutable**. No code shall modify, overwrite, or append to it.
  - The EDA notebook is **display-only**: it loads the raw data, prints `head()`, `info()`, `describe()`, null-checks, and generates descriptive visualizations (histograms, boxplots, pairplots, correlation heatmap).
  - A `# READ-ONLY` marker comment is required at the top of the notebook.
  - Outputs (plots) may be saved to `results/plots/` but the raw data must not be altered in any way.

### Stage 1 — Consolidated Data Preparation *(ALL PREPROCESSING)*

- **Owner:** `src/data_prep.py`
- **Input:** `data/raw/50_Startups.csv`
- **Output:** `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- **Operations (all in this single module):**
  1. **Load** raw CSV into a pandas DataFrame.
  2. **One-Hot Encoding** on the categorical `State` column (`pd.get_dummies` or `OneHotEncoder`).
  3. **Handle Dummy Variable Trap** — drop exactly one dummy column (e.g., `State_California` or `State_Florida`) to avoid multicollinearity.
  4. **Feature Matrix & Target Vector** — separate `Profit` (target) from all other columns (features).
  5. **Train-Test Split** — apply `train_test_split` with a fixed `random_state` (e.g., 42) and a configurable `test_size` (default 0.2).
  6. **Feature Scaling** — fit `StandardScaler` on `X_train` only; transform both `X_train` and `X_test` using the fitted scaler. **Never fit on test data.**
  7. **Save** the resulting NumPy arrays / DataFrames to `data/processed/`.

- **Key Rule:** No preprocessing logic (encoding, scaling, splitting) is permitted in any other file. Everything lives in `data_prep.py`.

### Stage 2 — Model Training

- **Owner:** `src/train.py`
- **Input:** `X_train`, `y_train` (loaded from `data/processed/`)
- **Process:**
  - Instantiate `LinearRegression()` from scikit-learn.
  - Fit the model on `X_train` and `y_train`.
  - Serialize the trained model with `joblib` or `pickle` → `models/linear_regression.pkl`.
  - (Optional) Also fit `statsmodels.OLS` for detailed summary statistics (p-values, confidence intervals).

### Stage 3 — Model Evaluation

- **Owner:** `src/evaluate.py`
- **Input:** Trained model + `X_test`, `y_test`
- **Process:**
  - Load the serialized model from `models/`.
  - Predict on `X_test`.
  - Compute metrics: **R²**, **Adjusted R²**, **RMSE**, **MAE**.
  - Write metrics to `results/metrics.json`.
  - Generate and save diagnostic plots:
    - Residuals vs. Fitted Values → `results/plots/residuals.png`
    - Actual vs. Predicted scatter → `results/plots/actual_vs_predicted.png`

---

## 4. Enforcement of Rules

| Rule                               | Mechanism                                                    |
| ---------------------------------- | ------------------------------------------------------------ |
| Raw data is read-only              | File permissions (read-only on OS), `# READ-ONLY` banner in notebook, no write paths targeting `data/raw/` in any `.py` file |
| All preprocessing is consolidated  | Only `src/data_prep.py` imports `OneHotEncoder` / `StandardScaler` / `train_test_split` for transformation purposes |
| EDA stage performs no transforms   | `notebooks/01_eda.ipynb` contains zero `pd.get_dummies`, `StandardScaler`, or `train_test_split` calls |
| No data leakage                    | Scalers / encoders are fitted **exclusively** on the training split; test split receives only `.transform()` |
| Reproducibility                    | All random operations use a fixed `random_state=42`          |

---

## 5. Data Dictionary — `50_Startups.csv`

| Column           | Type     | Description                                            |
| ---------------- | -------- | ------------------------------------------------------ |
| `R&D Spend`      | float64  | Money spent on Research & Development (USD)            |
| `Administration` | float64  | Money spent on Administration (USD)                    |
| `Marketing Spend`| float64  | Money spent on Marketing (USD)                         |
| `State`          | object   | Startup location: `New York`, `California`, `Florida` |
| `Profit`         | float64  | **Target variable** — Profit earned (USD)             |

---

## 6. Execution Order

```bash
# 1. Explore data (read-only)
jupyter notebook notebooks/01_eda.ipynb

# 2. Preprocess + split + scale (all-in-one)
python src/data_prep.py

# 3. Train model
python src/train.py

# 4. Evaluate model
python src/evaluate.py
```

---

## 7. Guardrails & Checklist

- [ ] `data/raw/` directory has read-only OS permissions.
- [ ] `notebooks/01_eda.ipynb` begins with `# READ-ONLY` comment.
- [ ] `src/data_prep.py` is the **only** file containing encoding, scaling, and splitting logic.
- [ ] `StandardScaler` is fit on `X_train` only — never on the full dataset or on `X_test`.
- [ ] Exactly one dummy variable is dropped to avoid the trap.
- [ ] All `random_state` values are set to `42`.
- [ ] Model is serialized before evaluation for reproducibility.
- [ ] No raw data file is ever written to by any pipeline script.

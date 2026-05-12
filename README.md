# 🇵🇰 Pakistan Freelancing Earnings Predictor — ML Pipeline

An end-to-end Machine Learning pipeline to predict monthly freelancer earnings in Pakistan using real economic data from 2015–2025.

## 📊 Dataset

[Pakistan Freelancing Economy 2015–2025](https://www.kaggle.com/datasets/hussnainmamoon1/pakistan-freelancing-economy-20152025) — 6,000 freelancer records

## 🚀 Results

| Metric | Score |
|--------|-------|
| R² Score | 0.82 |
| RMSE | Rs. 126,546 PKR |
| Algorithm | Linear Regression |

## 🔧 Complete Pipeline

### 1. EDA (Exploratory Data Analysis)
- Checked shape, dtypes, nulls, duplicates
- Used `df.describe()` to spot outliers
- Checked min/max values against real-world logic

### 2. Feature Selection
- Removed flag columns (age_missing_flag, edu_missing_flag etc.)
- Removed irrelevant identifiers (freelancer_id)
- Dropped high cardinality columns (city, primary_skill)
- Dropped collinear columns (monthly_earnings_usd, pkr_usd_rate)

### 3. Outlier Detection & Handling
- IQR method → statistical outliers
- Domain knowledge → impossible values (negative earnings, negative experience)
- Fixed using `clip(lower=0, upper=...)` method

### 4. Null Handling
- review_score → 6.2% nulls → filled with median ✅

### 5. Feature Engineering
- **map()** → ordered categories (education_level, english_proficiency)
- **LabelEncoder** → binary columns (gender, freelancing_as_primary_income)
- **One-Hot Encoding** → unordered categories (platform, skill_category etc.)

### 6. Scaling
- `StandardScaler` applied to all numerical features

### 7. Train/Test Split
- 80% training / 20% testing
- `random_state=42` for reproducibility

### 8. Model Training
- `LinearRegression` from sklearn
- Target: `monthly_earnings_pkr`

### 9. Evaluation & Visualization
- R² Score, MSE, RMSE
- Actual vs Predicted scatter plot
- Error Distribution histogram

## 📈 Key Insights

```
✅ Model predicts low earners accurately
❌ Struggles with high earners (data imbalance)
✅ Error distribution centered at 0 → unbiased model
```

## 💡 Lessons Learned

- Always verify `map()` keys match actual dataset values exactly
- Negative predicted values need clipping (earnings can't be negative!)
- RMSE in original unit (PKR) is far more readable than raw MSE
- Domain knowledge catches outliers that IQR misses

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)

## 📁 Files

```
├── README.md
└── Testing.ipynb   ← main notebook
```

## 🔜 Next Steps

- Model Tuning (Hyperparameter Optimization)
- Cross Validation
- Try Random Forest / Gradient Boosting for better R²

## 👤 Author

**Talha** — Final Year CS Student @ NUML  
[LinkedIn](https://www.linkedin.com/in/) | [GitHub](https://github.com/)

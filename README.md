# Diabetes Prediction: Data Mining Project

This repo holds my part (Fairuz) of our group's data mining project on diabetes prediction.

🌐 **Full group project website:** https://sites.google.com/view/diabetes-gigabyte/home

## What's in this repo

| File | Description |
|------|-------------|
| [`diabetes_prediction.ipynb`](diabetes_prediction.ipynb) | Full notebook: preprocessing, feature engineering, classification, clustering, association rules and explainable AI |
| `requirements.txt` | Python packages used |

## Datasets

The notebook uses two datasets (not included in this repo):

- **Dataset 1, "Big" (survey data):** `diabetes_big.csv`, a health-indicator survey dataset with `Diabetes_binary`, `HighBP`, `BMI`, `Sex` and `Age` (age category 1–13)
- **Dataset 2, "Small" (Pima):** `pima_small.csv`, the Pima Indians Diabetes dataset (`Glucose`, `BloodPressure`, `BMI`, `Age`, `Outcome`, …)

Models are **trained on Dataset 1 and tested on Dataset 2**, so they're evaluated on a separate population.

## Workflow

1. **Preprocessing**
   - Dataset 1: BMI outliers capped at the 99th percentile; age categories mapped to approximate ages
   - Dataset 2: zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` treated as missing and filled with the median
   - Both datasets mapped to a common schema: `Diabetes_binary`, `HighBP`, `BMI`, `Sex`, `Age_Raw`
2. **Feature engineering:** `Age_Normalized`, `Metabolic_Score` (BMI × (HighBP + 1)), `Age_Weighted_BMI`, `Age_HighBP`
3. **Classification** (GridSearchCV tuning, F1 scoring): Decision Tree, Random Forest, XGBoost, Logistic Regression, kNN, SVM, Gaussian Naive Bayes, ANN (MLP)
4. **Explainable AI:** XGBoost feature importance and SHAP
5. **Clustering:** k-Means, DBSCAN, Hierarchical (silhouette score, PCA 2D visualisation, centroid interpretation)
6. **Association rules:** Apriori and FP-Growth with support tuning

## Results (summary)

**Classification (test set = Dataset 2)**

| Algorithm | Accuracy | F1 Score |
|-----------|---------:|---------:|
| Decision Tree | 0.678 | 0.355 |
| Random Forest | 0.680 | 0.369 |
| XGBoost | 0.682 | 0.361 |
| Logistic Regression | 0.668 | 0.358 |
| **kNN** | **0.686** | **0.377** |
| SVM | 0.673 | 0.351 |
| Gaussian NB | 0.676 | 0.332 |
| ANN (MLP) | 0.672 | 0.364 |

**Clustering (silhouette score):** k-Means 0.414 · Hierarchical 0.414 · DBSCAN 0.283

**Association rules:** Apriori and FP-Growth both found 22 rules at support 0.2 (avg. lift 1.35, avg. confidence 0.62).

## How to run

The notebook was written for **Google Colab** and reads the CSVs from Google Drive:

```python
df1 = pd.read_csv('/content/drive/MyDrive/datasets/diabetes_big.csv')
df2 = pd.read_csv('/content/drive/MyDrive/datasets/pima_small.csv')
```

To run it:

1. Open the notebook in Colab (or locally with Jupyter).
2. Put both CSV files in `MyDrive/datasets/`, or change the paths above to where your files are.
3. If you run it locally, install the dependencies with `pip install -r requirements.txt` and remove the `google.colab` drive-mount cell.

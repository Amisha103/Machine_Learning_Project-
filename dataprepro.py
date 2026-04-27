import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings 

warnings.filterwarnings('ignore')

# ===================== LOAD DATA =====================
df = pd.read_csv('insurance.csv')

# ===================== EDA =====================
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# Check nulls
print("\nNull values:\n", df.isnull().sum())

# ===================== DATA CLEANING =====================
df_cleaned = df.copy()

print("\nBefore removing duplicates:", df_cleaned.shape)
df_cleaned.drop_duplicates(inplace=True)
print("After removing duplicates:", df_cleaned.shape)

print("\nData types:\n", df_cleaned.dtypes)
print("\nSex value counts:\n", df_cleaned['sex'].value_counts())

# Encode categorical (binary)
df_cleaned['sex'] = df_cleaned['sex'].map({"male": 0, "female": 1})
df_cleaned['smoker'] = df_cleaned['smoker'].map({"no": 0, "yes": 1})

# Rename columns (NO inplace assignment bug)
df_cleaned = df_cleaned.rename(columns={
    'sex': 'is_female',
    'smoker': 'is_smoker'
})

# Region encoding
print("\nRegion counts:\n", df_cleaned['region'].value_counts())
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)

# ===================== FEATURE ENGINEERING =====================
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)

# ===================== FEATURE SCALING =====================
from sklearn.preprocessing import StandardScaler

cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])

print("\nProcessed Data Sample:\n", df_cleaned.head())

# ===================== CORRELATION =====================
from scipy.stats import pearsonr

# Take all features except target
selected_features = [col for col in df_cleaned.columns if col != 'charges']

correlations = {}

for feature in selected_features:
    try:
        corr, _ = pearsonr(df_cleaned[feature], df_cleaned['charges'])
        correlations[feature] = corr
    except:
        correlations[feature] = np.nan

correlation_df = pd.DataFrame(
    list(correlations.items()),
    columns=['Feature', 'Pearson Correlation']
)

correlation_df = correlation_df.sort_values(
    by='Pearson Correlation',
    ascending=False
)

print("\nFeature Correlations with Charges:\n")
print(correlation_df)

cat_features = ['is_female', 'is_smoker', 'region_northwest', 'region_southeast', 'region_southwest','bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese']

from scipy.stats import chi2_contingency
alpha = 0.05
df_cleaned['charges_bins'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)
chi2_results = {}
for  col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bins'])
    chi2_stat, p_val,_, _ = chi2_contingency(contingency)
    decision  = "reject Null(keep feature)" if p_val < alpha else "Accept Null(drop feature)"
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'decision': decision
    }
chi2_df = pd.DataFrame.from_dict(chi2_results, orient='index')
print("\nChi-Square Test Results:\n")
print(chi2_df)
final_df = df_cleaned[['age', 'is_female', 'bmi', 'children', 'is_smoker', 'region_southeast', 'bmi_category_Obese','charges']]
print("\nFinal Data Sample:\n", final_df.head())
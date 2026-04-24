import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings 

warnings.filterwarnings('ignore')

df = pd.read_csv('insurance.csv')
#EDA

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

#checking for null values

print(df.isnull().sum())

numeric_columns = ['age', 'bmi', 'children', 'charges']

# for col in numeric_columns:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(df[col], kde = True, bins = 20)
#     plt.show()

# sns.countplot(df['sex'])
# plt.show()

# sns.countplot(x = df['smoker'])
# plt.show()

# for col in numeric_columns:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x = df[col])
#     plt.show()

# plt.figure(figsize = (8, 6))
# sns.heatmap(df.corr(numeric_only=True), annot = True)
# plt.show()

#DATA CLEANING AND PROCESSING
df_cleaned = df.copy()
print(df_cleaned.shape)
df_cleaned.drop_duplicates(inplace=True)
print(df_cleaned.shape)
print(df_cleaned.isnull().sum())
print(df_cleaned.dtypes)
print(df_cleaned['sex'].value_counts())

df_cleaned['sex'] = df_cleaned['sex'].map({"male":0, "female":1})


df_cleaned['smoker'].value_counts()
df_cleaned['smoker'] = df_cleaned['smoker'].map({"no": 0, "yes":1})
df_cleaned = df_cleaned.rename(columns={'sex':'is_female', 'smoker':'is_smoker'}, inplace = True)
print(df_cleaned.head())
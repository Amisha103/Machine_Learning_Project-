import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')
print("Shape:", df.shape)
print(df.head())

#####EDA#####
print(df.columns)
print(df.info())
print(df.describe()) #only for numerical data
print("\nNull values:\n", df.isnull().sum())
print("\nDuplicated values:\n", df.duplicated().sum())
print(df['HeartDisease'].value_counts())
# df['HeartDisease'].value_counts().plot(kind='bar')
# plt.show()

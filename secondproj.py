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

# def plotting(var, num):
#     plt.subplot(2, 2, num)
#     sns.histplot(df[var], kde=True)
#     plt.show()
    

# plotting('Age', 1)
# plotting('RestingBP', 2)
# plotting('Cholesterol', 3)
# plotting('MaxHR', 4)
# plt.tight_layout()


ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
df['Cholestrol'] = df['Cholesterol'].replace(0, ch_mean)
df['Cholesterol'] = df['Cholestrol'].round(2)

ch_restingbp = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
df['RestingBP'] = df['RestingBP'].replace(0, ch_restingbp)
df['RestingBP'] = df['RestingBP'].round(2)

def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)
    plt.show()
    

plotting('Age', 1)
plotting('RestingBP', 2)
plotting('Cholesterol', 3)
plotting('MaxHR', 4)
plt.tight_layout()

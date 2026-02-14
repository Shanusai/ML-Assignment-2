# EDA and preprocessing for Customer Churn dataset (human-style)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('--- TRAIN INFO ---')
print(train.info())
print('\n--- TRAIN DESCRIBE ---')
print(train.describe())
print('\n--- CHURN VALUE COUNTS ---')
print(train['Churn'].value_counts())

sns.countplot(x='Churn', data=train)
plt.title('Churn Class Distribution')
plt.savefig('eda_churn_distribution.png')
plt.close()

corr = train.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('eda_corr_heatmap.png')
plt.close()

print('EDA plots saved as eda_churn_distribution.png and eda_corr_heatmap.png')

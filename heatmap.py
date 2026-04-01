import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("final_data.csv")

numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='coolwarm')
plt.show()

print(corr['price'].sort_values(ascending=False))
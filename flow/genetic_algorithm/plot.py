import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os

#df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
#ax = df.plot.bar(x='lab', y='val', rot=0)
file = "/home/macsilva/Desktop/maslab/flow/flow/genetic_algorithm/csv/values/home.csv"
df = pd.read_csv(file)
max_df = df.loc[df["generation"] <= 16].groupby(["generation"]).max()
min_df = df.loc[df["generation"] <= 16].groupby(["generation"]).min()
mean_df = df.loc[df["generation"] <= 16].groupby(["generation"]).mean()
sns.boxplot(data=df.loc[df["generation"] <= 16], x="generation", y="value")
plt.show()
print(max_df)
print(min_df)
print(mean_df)


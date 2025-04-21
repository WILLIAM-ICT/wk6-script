# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore dataset structure
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())

# Grouping by species and computing mean of numerical columns
print("\nMean values grouped by species:")
print(df.groupby('target').mean())

# Rename target column with species names for clarity
df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))

# Task 3: Data Visualization

# Set seaborn style
sns.set(style="whitegrid")

# 1. Line chart: example using sepal length mean across species
mean_values = df.groupby('species')['sepal length (cm)'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.lineplot(data=mean_values, x='species', y='sepal length (cm)', marker='o')
plt.title('Average Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.tight_layout()
plt.show()

# 2. Bar chart: average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='petal length (cm)', estimator='mean', ci=None)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram: distribution of sepal width
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot: sepal length vs. petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs. Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()
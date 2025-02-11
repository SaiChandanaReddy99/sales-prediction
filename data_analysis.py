import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = "Dataset/Cleaned_Advertising_Data.csv"
df = pd.read_csv(file_path)

# Display summary statistics
print("Summary Statistics:")
print(df.describe())

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot to visualize relationships
sns.pairplot(df)
plt.show()

# Distribution Plots
for col in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Scatter Plots for Budget vs Sales
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(x=df['TV_Budget'], y=df['Sales'], ax=axes[0])
axes[0].set_title("TV Budget vs Sales")

sns.scatterplot(x=df['Radio_Budget'], y=df['Sales'], ax=axes[1])
axes[1].set_title("Radio Budget vs Sales")

sns.scatterplot(x=df['Newspaper_Budget'], y=df['Sales'], ax=axes[2])
axes[2].set_title("Newspaper Budget vs Sales")

plt.show()

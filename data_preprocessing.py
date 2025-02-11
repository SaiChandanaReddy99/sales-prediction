import pandas as pd

# Load the dataset
df = pd.read_csv("Dataset/Advertising Budget and Sales.csv")

# Drop the unnecessary index column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Rename columns for consistency
df.rename(columns={
    'TV Ad Budget ($)': 'TV_Budget',
    'Radio Ad Budget ($)': 'Radio_Budget',
    'Newspaper Ad Budget ($)': 'Newspaper_Budget',
    'Sales ($)': 'Sales'
}, inplace=True)

# Check for duplicate rows
df.drop_duplicates(inplace=True)

# Check for outliers using IQR method
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Apply outlier removal to all numerical columns
for col in ['TV_Budget', 'Radio_Budget', 'Newspaper_Budget', 'Sales']:
    df = remove_outliers(df, col)

# Save the cleaned dataset
df.to_csv("dataset/Cleaned_Advertising_Data.csv", index=False)

# Display cleaned dataset
df.head()

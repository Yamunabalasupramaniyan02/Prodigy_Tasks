import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Age': [23, 35, 45, 25, 67, 34, 50, 29, 38, 40]
}

df = pd.DataFrame(data)

# Bar chart for gender distribution
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 6))
gender_counts.plot(kind='bar', color=['blue', 'pink'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Histogram for age distribution
plt.figure(figsize=(8, 6))
plt.hist(df['Age'], bins=5, color='green', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

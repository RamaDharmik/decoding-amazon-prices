import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(r"C:\Users\ramad\Music\projects\amazon.csv")

# Clean price & discount columns
df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
df['discount_percentage'] = df['discount_percentage'].replace('%', '', regex=True).astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

# Drop unnecessary columns
df.drop(columns=['img_link', 'product_link', 'about_product'], inplace=True)

# Basic data checks
df.info()
df.head()
df.shape
df.isnull().sum()
df.describe()
df['category'].value_counts().head(10)

# ---------- 📊 Figure 1: Top 10 Product Categories ----------
plt.figure(figsize=(10,6))
sns.countplot(y='category', data=df, order=df['category'].value_counts().head(10).index)
plt.title("Figure 1: Top 10 Product Categories")
plt.xlabel("Number of Products")
plt.ylabel("Category")
plt.show()

# ---------- 🔝 Top Rated Products ----------
top_rated = df[(df['rating'] >= 4.5) & (df['rating_count'] >= 100)]
top_rated[['product_name', 'category', 'rating', 'rating_count']].sort_values(by='rating', ascending=False).head(10)

# ---------- 🔝 Most Reviewed Products ----------
df[['product_name', 'category', 'rating_count']].sort_values(by='rating_count', ascending=False).head(10)

# ---------- 🔝 Highest Discounted Products ----------
df[['product_name', 'category', 'actual_price', 'discounted_price', 'discount_percentage']].sort_values(by='discount_percentage', ascending=False).head(10)

# ---------- 📊 Figure 2: Price vs Discount Percentage (Fixed & Improved) ----------
# Add jitter to avoid overlapping
df['discount_jitter'] = df['discount_percentage'] + np.random.uniform(-1, 1, size=len(df))

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df, 
    x='actual_price', 
    y='discount_jitter', 
    hue='discount_percentage', 
    size='discount_percentage', 
    sizes=(20, 200),
    palette='coolwarm',
    alpha=0.7,
    legend=False
)
plt.title("Figure 2: Actual Price vs Discount Percentage (with Jitter)")
plt.xlabel("Actual Price (₹)")
plt.ylabel("Discount Percentage")
plt.grid(True)
plt.show()

# ---------- 📊 Figure 3: Top 10 Categories by Average Rating ----------
category_rating = df.groupby('category')['rating'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=category_rating.values, y=category_rating.index, palette='viridis', hue=category_rating.index)
plt.legend([],[], frameon=False)
plt.title("Figure 3: Top 10 Categories by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Category")
plt.show()

# ---------- 📊 Figure 4: Correlation Heatmap ----------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Figure 4: Correlation Heatmap")
plt.show()

# ---------- ✅ Save Cleaned Data ----------
df.to_csv("cleaned_amazon_data.csv", index=False)

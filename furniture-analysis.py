# 1]Data Collection

# Import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle



# Load Datasets 
furniture=pd.read_csv("ecommerce_furniture_dataset.csv")

# Check basic info
print("\nDataset Info:")
print(furniture.info())

# Viewing first 5 rows
print("First 5 rows of the dataset:")
print(furniture.head())

# 2] Data Cleaning 

# Check for missing values
print("\nMissing values:")
print(furniture.isnull().sum())

# Drop missing values
furniture.dropna(inplace=True)

# Check for duplicates
print("\nDuplicate values:")
print(furniture.duplicated().sum())

# Drop duplicates
furniture.drop_duplicates(inplace=True)

# Drop 'originalPrice' column if it has too many missing values
furniture.dropna(subset=['originalPrice', 'price'], inplace=True)

# Clean the 'originalPrice' column
furniture['originalPrice'] = furniture['originalPrice'].replace('[\$,]', '', regex=True)
furniture['originalPrice'] = pd.to_numeric(furniture['originalPrice'], errors='coerce')

# Clean the 'price' column
furniture['price'] = furniture['price'].replace('[\$,]', '', regex=True)
furniture['price'] = pd.to_numeric(furniture['price'], errors='coerce')

# Clean the 'tagText' column (categorize values)
furniture['tagText'] = furniture['tagText'].apply(lambda x: x if x in ['Free shipping', '+Shipping: $5.09'] else 'others')

# Encode 'tagText'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
furniture['tagText'] = le.fit_transform(furniture['tagText'])

# Final check
print(furniture.head())


# 3] Exploratory Data Analysis

# Plot price distribution
sns.histplot(furniture['price'], kde=True)
plt.title("Distribution of Item Prices")
plt.show()

# Plot sold distribution
sns.histplot(furniture['sold'], kde=True)
plt.title("Distribution of Items Sold")
plt.show()

# Price vs Sold scatter
sns.scatterplot(x='price', y='sold', data=furniture)
plt.title("Price vs Items Sold")
plt.show()

# Relationship between originalPrice, price and sold
sns.pairplot(furniture, vars=['originalPrice', 'price', 'sold'],kind='scatter')
plt.title('Relationship Between Price, Original Price, and Items Sold')
plt.show()

# 4] Feature Engineering 

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a new feature: percentage discount
furniture['discount_percentage'] = ((furniture['originalPrice'] - furniture['price']) / furniture['originalPrice']) * 100

# Fill missing values in productTitle
furniture['productTitle'].fillna('', inplace=True)

# Convert productTitle into numeric features using TF-IDF
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
productTitle_tfidf = tfidf.fit_transform(furniture['productTitle'])

# Convert to DataFrame and concatenate with original DataFrame
productTitle_tfidf_df = pd.DataFrame(productTitle_tfidf.toarray(), columns=tfidf.get_feature_names_out())
productTitle_tfidf_df.reset_index(drop=True, inplace=True)
furniture.reset_index(drop=True, inplace=True)
furniture = pd.concat([furniture, productTitle_tfidf_df], axis=1)

# Drop original productTitle as it's now encoded
furniture.drop(columns=['productTitle'], inplace=True)

# 5] Model Selection And training 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Split dataset into features and target
X = furniture.drop('sold', axis=1)
y = furniture['sold']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# 6] Model Evaluation

from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression Predictions
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest Predictions
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print Evaluation Results
print(f"Linear Regression -> MSE: {mse_lr:.2f}, R2: {r2_lr:.2f}")
print(f"Random Forest     -> MSE: {mse_rf:.2f}, R2: {r2_rf:.2f}")

# Conclusion
if r2_rf > r2_lr:
    print("Random Forest performed better in predicting furniture sales.")
else:
    print("Linear Regression performed better, indicating a more linear relationship.")

# Model Saving 
# Save Best Model
best_model = rf_model if r2_rf > r2_lr else lr_model

with open("furniture_sales_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

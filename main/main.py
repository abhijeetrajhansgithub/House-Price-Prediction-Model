import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model.LinearRegressionModel_ import LinearRegressionModel
from model.SupportVectorRegression_ import SupportVectorRegressionModel
from model.DecisionTreeRegression_ import DecisionTreeRegressionModel
from model.RandomForestRegression_ import RandomForestRegressionModel
from model.AdaBoostRegression_ import AdaBoostRegressionModel
from model.XGBoostRegression_ import XGBoostRegressionModel
from model.GradientBoostingRegression_ import GradientBoostingRegressionModel
from model.LightGBMRegression_ import LightGBMRegressionModel
from model.KNNRegression_ import KNNRegressionModel
from model.NeuralNetworkRegression_ import MLPRegressionModel

# Load the data
data = pd.read_csv(r"..\data\data.csv")
print(data.head())
print(data.info())

data = data[data["price"] != 0]

# one hot encoding for city
categorical_columns = ['city']
label_encoder = LabelEncoder()

# Fit the label encoder and transform the labels to numerical values
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Convert the price column to float
data['price'] = data['price'].astype(float)

# Drop rows with missing values
data.dropna(inplace=True)
print(data.head())

# Split the dataset into training and testing sets
X = data.drop(['price', data.columns[0]], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Visualize the distribution of house prices
plt.figure(figsize=(20, 20))
train_data.hist(figsize=(20, 20), bins=50, xlabelsize=10, ylabelsize=10)
plt.xlabel('Price')
plt.ylabel('Count of House')
plt.title('Distribution of House Prices')
plt.show()

# Correlation heatmap
train_corr_data_columns = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
                           "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated"]
plt.figure(figsize=(20, 20))
sns.heatmap(train_data[train_corr_data_columns].corr(), annot=True, cmap="YlGnBu", annot_kws={"size": 10})
plt.show()

# Log transformation of skewed features
log_transform_columns = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "sqft_basement",
                         "yr_built", "yr_renovated"]
for col in log_transform_columns:
    train_data[col] = np.log(train_data[col] + 1)
    test_data[col] = np.log(test_data[col] + 1)

# Visualize the distribution again after log transformation
plt.figure(figsize=(20, 20))
train_data.hist(figsize=(20, 20), bins=50, xlabelsize=10, ylabelsize=10)
plt.xlabel('Price')
plt.ylabel('Count of House')
plt.title('Distribution of House Prices (Log Transformed)')
plt.show()

print(train_data["city"].value_counts())

# Remove "statezip"
train_data.drop(["statezip"], axis=1, inplace=True)
test_data.drop(["statezip"], axis=1, inplace=True)

# # One Hot Encoding of "city"
# train_data = pd.get_dummies(train_data, columns=["city"])
# test_data = pd.get_dummies(test_data, columns=["city"])

# Add bathroom ratio
train_data["bathroom_ratio"] = train_data["bathrooms"] / train_data["bedrooms"]
test_data["bathroom_ratio"] = test_data["bathrooms"] / test_data["bedrooms"]

# Synchronize columns between train and test data
for column in train_data.columns:
    if column not in test_data.columns:
        test_data[column] = 0

for column in test_data.columns:
    if column not in train_data.columns:
        train_data[column] = 0

# Drop unnecessary columns
columns_to_drop = ["country", "street"]
train_data.drop(columns_to_drop, axis=1, inplace=True)
test_data.drop(columns_to_drop, axis=1, inplace=True)

# Ensure column order is the same
train_data = train_data[test_data.columns]

print("Train data\n", train_data.head(), "\nTest data\n", test_data.head())
print("Number of columns in train data:", len(train_data.columns))
print("Number of columns in test data:", len(test_data.columns))
print("Columns in train data:", train_data.columns)
print("Columns in test data:", test_data.columns)

# Print bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, yr_built, yr_renovated and price of the first five rows in both train and test data
columns_to_show = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "sqft_basement",
                   "yr_built", "yr_renovated", "price"]

print(train_data[columns_to_show].head())
print(test_data[columns_to_show].head())

# Train the model
print("\nLinear Regression Model")
LinearRegressionModel(train_data, test_data)

# Train the model
print("\nSupport Vector Regression Model")
SupportVectorRegressionModel(train_data, test_data)

# Train the model
print("\nDecision Tree Regression Model")
DecisionTreeRegressionModel(train_data, test_data)

# Train the model
print("\nRandom Forest Regression Model")
RandomForestRegressionModel(train_data, test_data)

# Train the model
print("\nAdaBoost Regression Model")
AdaBoostRegressionModel(train_data, test_data)

# Train the model
print("\nXGBoost Regression Model")
XGBoostRegressionModel(train_data, test_data)

# Train the model
print("\nGradient Boosting Regression Model")
GradientBoostingRegressionModel(train_data, test_data)

# Train the model
print("\nLightGBM Regression Model")
LightGBMRegressionModel(train_data, test_data)

# Train the model
print("\nKNN Regression Model")
KNNRegressionModel(train_data, test_data)

# Train the model
print("\nNeural Network Regression Model")
MLPRegressionModel(train_data, test_data)


def find_best_model():
    linear_regression = LinearRegressionModel(train_data, test_data)
    support_vector_regression = SupportVectorRegressionModel(train_data, test_data)
    decision_tree_regression = DecisionTreeRegressionModel(train_data, test_data)
    random_forest_regression = RandomForestRegressionModel(train_data, test_data)
    ada_boost_regression = AdaBoostRegressionModel(train_data, test_data)
    xg_boost_regression = XGBoostRegressionModel(train_data, test_data)
    gradient_boosting_regression = GradientBoostingRegressionModel(train_data, test_data)
    light_gbm_regression = LightGBMRegressionModel(train_data, test_data)
    knn_regression = KNNRegressionModel(train_data, test_data)
    neural_network_regression = MLPRegressionModel(train_data, test_data)

    # Find the best model
    best_model = max(linear_regression, support_vector_regression, decision_tree_regression, random_forest_regression,
                     ada_boost_regression, xg_boost_regression, gradient_boosting_regression, light_gbm_regression,
                     knn_regression, neural_network_regression)

    # Print the best model
    print("\n\nBest model:", best_model)


find_best_model()

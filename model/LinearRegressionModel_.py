from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def LinearRegressionModel(train_data, test_data):
    # Fill missing values with the mean of each column
    train_data = train_data.fillna(train_data.mean())
    test_data = test_data.fillna(test_data.mean())

    # Ensure columns are in the same order
    train_columns = train_data.columns
    test_data = test_data[train_columns]

    print("==================================================================")
    print("Train data\n", train_data.head(), "\nTest data\n", test_data.head())

    # Separate features and target variable
    x_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    x_test = test_data.drop('price', axis=1)
    y_test = test_data['price']

    # Train the linear regression model
    reg = LinearRegression().fit(x_train, y_train)

    # Predict on the test set
    y_pred = reg.predict(x_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    # Save the model
    import pickle
    with open('../saved_models/linear_regression_model.pkl', 'wb+') as f:
        pickle.dump(reg, f)

    return r2

# Example usage (assuming train_data and test_data are already defined):
# model = LinearRegressionModel(train_data, test_data)

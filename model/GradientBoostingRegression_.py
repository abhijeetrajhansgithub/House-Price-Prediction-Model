# Gradient Boosting Regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def GradientBoostingRegressionModel(train_data, test_data):
    # Fill missing values with the mean of each column
    train_data = train_data.fillna(train_data.mean())
    test_data = test_data.fillna(test_data.mean())

    # Ensure columns are in the same order
    train_columns = train_data.columns
    test_data = test_data[train_columns]

    # Separate features and target variable
    x_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    x_test = test_data.drop('price', axis=1)
    y_test = test_data['price']

    # Train the Gradient Boosting regression model
    gbr = GradientBoostingRegressor()
    gbr.fit(x_train, y_train)

    # Predict on the test set
    y_pred = gbr.predict(x_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

    # Save the model
    import pickle
    with open('../saved_models/gbr_model.pkl', 'wb+') as f:
        pickle.dump(gbr, f)

    return r2

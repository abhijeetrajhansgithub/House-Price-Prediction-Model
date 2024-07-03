# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def DecisionTreeRegressionModel(train_data, test_data):
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

    # Train the decision tree regression model
    dtr = DecisionTreeRegressor()
    dtr.fit(x_train, y_train)

    # Predict on the test set
    y_pred = dtr.predict(x_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", mse**0.5)
    print("R-squared:", r2_score(y_test, y_pred))

    # Save the model
    import pickle
    with open('../saved_models/dtr_model.pkl', 'wb+') as f:
        pickle.dump(dtr, f)

    return r2_score(y_test, y_pred)

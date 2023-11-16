### Name : ADVAIT GURUNATH CHAVAN
### Contact Number : +91 70214 55852
### Email ID : advaitchavan135@gmail.com 
### Forage Cognizant Artificial Intelligence Task 4 : Machine Learning Production

# Importing the necessary modules
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("Cleaned and combined dataset.csv")

# Number of folds for K-fold cross-validation
K = 10
    
# Separate features (x) and target variable (y)
x = data.drop(columns=['estimated_stock_pct'])
y = data['estimated_stock_pct']

# Ratio for splitting data into training and test sets
split = 0.75

# List to store accuracy scores during cross-validation
accuracy = []

# Instantiate a StandardScaler to standardize feature values
scaler = StandardScaler()

# Create training and test samples using the specified split ratio and random seed
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split, random_state=42)

# Fit the scaler on the training data and transform both training and test data
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Assigning variable to the models
model_1 = RandomForestRegressor()
model_2 = DecisionTreeRegressor()

def train_and_test_RandomForestRegressor():
    print("Training and testing using RandomForestRegressor Model: -")
    
    # Loop through each fold in K-fold cross-validation
    for fold in range(0, K):
    
        # Train the RandomForestRegressor model
        trained_model_1 = model_1.fit(x_train, y_train)
    
        # Generate predictions on the test sample
        y_pred = trained_model_1.predict(x_test)
    
        # Compute accuracy using mean absolute error (MAE)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
        # Append the MAE to the accuracy list
        accuracy.append(mae)
    
        # Print the MAE for the current fold
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")
    
    # Calculate and print the average MAE across all folds
    print(f"Average MAE for RandomForestRegressor Model: {(sum(accuracy) / len(accuracy)):.2f}")

def evaluation_RandomForestRegressor():
    print("Top-5 features and their relative-importance for predicting the target variable(estimated_stock_pct) using RandomForestRegressor Model: -")
    """
    Evaluate and print the top 5 features and their importances from a RandomForestRegressor model.

    Parameters:
    - model: RandomForestRegressor model trained on the dataset.
    - x: DataFrame containing features.

    Returns:
    None
    """
    features_1 = [i.split("__")[0] for i in x.columns]
    importances_1 = model_1.feature_importances_
    sorted_features_1 = sorted(zip(features_1, importances_1), key=lambda x: x[1], reverse=True)

    # Print the top 5 features and their importances
    top_features_1 = sorted_features_1[:5]
    for feature, importance in top_features_1:
        print(f"{feature} : {round(importance, 3)}")
    
def train_and_test_DecisionTreeRegressor():
    print("Training and testing using DecisionTreeRegressor Model: -")
    
    # Loop through each fold in K-fold cross-validation
    for fold in range(0, K):
    
        # Train the DecisionTreeRegressor model
        trained_model_2 = model_2.fit(x_train, y_train)
    
        # Generate predictions on the test sample
        y_pred = trained_model_2.predict(x_test)
    
        # Compute accuracy using mean absolute error (MAE)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
        # Append the MAE to the accuracy list
        accuracy.append(mae)
    
        # Print the MAE for the current fold
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")
    
    # Calculate and print the average MAE across all folds
    print(f"Average MAE for DecisionTreeRegressor Model: {(sum(accuracy) / len(accuracy)):.2f}")


def evaluation_DecisionTreeRegressor():
    print("Top-5 features and their relative-importance for predicting the target variable(estimated_stock_pct) using DecisionTreeRegressor Model: -")
    """
    Evaluate and print the top 5 features and their importances from a DecisionTreeRegressor model.

    Parameters:
    - model: DecisionTreeRegressor model trained on the dataset.
    - x: DataFrame containing features.

    Returns:
    None
    """
    features_2 = [i.split("__")[0] for i in x.columns]
    importances_2 = model_2.feature_importances_
    sorted_features_2 = sorted(zip(features_2, importances_2), key=lambda x: x[1], reverse=True)

    # Print the top 5 features and their importances
    top_features_2 = sorted_features_2[:5]
    for feature, importance in top_features_2:
        print(f"{feature} : {round(importance, 3)}")
   


train_and_test_RandomForestRegressor()
evaluation_RandomForestRegressor()
train_and_test_DecisionTreeRegressor()
evaluation_DecisionTreeRegressor()
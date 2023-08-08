# %% [markdown]
# # XGBoost Classification on Titanic Dataset
# **XGBoost (Extreme Gradient Boosting) is a powerful algorithm widely used in machine learning, known for its high performance in various applications. In this article, we will use XGBoost to perform classification on the famous Titanic dataset. We'll walk through the steps of data preprocessing, model training, cross-validation, and result evaluation in Python.**
# 

# %% [markdown]
# ## Titanic Dataset
# **The Titanic dataset contains information about passengers who traveled on the RMS Titanic, including whether they survived or not. Our goal is to build a classification model that predicts the survival of passengers.**
# 
# 

# %% [markdown]
# ## Step 1: Data Loading and Preprocessing
# **Let's start by loading the dataset and performing some preprocessing steps:**

# %%
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
data = pd.read_csv("titanic.csv")


# %%
data.head()

# %%
# Data preprocessing
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data = pd.get_dummies(data, columns=['Embarked', 'Pclass'], drop_first=True)

# Separate independent variables and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']

# %% [markdown]
# ## Step 2: Splitting the Data into Training and Testing Sets
# **Let's split our dataset into training and testing sets:**

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# ## Step 3: Creating the XGBoost Model
# 
# **Before creating the XGBoost model, let's set some hyperparameters:**
# 

# %%
dtrain = xgb.DMatrix(X_train, label=y_train)

# Set hyperparameters
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}


# %% [markdown]
# ## Step 4: Training the Model with Cross-Validation
# **Cross-validation is used to evaluate the model's performance more reliably. In this step, we'll find the best model using cross-validation:**

# %%
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    nfold=5,
    metrics=['error'],
    early_stopping_rounds=10,
    stratified=True,
    seed=42
)

best_iteration = cv_results['test-error-mean'].idxmin()
best_error = cv_results.loc[best_iteration, 'test-error-mean']

print(f"Best error rate: {best_error:.4f} (iteration: {best_iteration+1})")


# %% [markdown]
# ## Step 5: Making Predictions on Test Data and Evaluating the Results
# **Finally, we'll select the best model and make predictions on the test data to evaluate its performance:**

# %%
dtest = xgb.DMatrix(X_test)

best_model = xgb.train(params, dtrain, num_boost_round=best_iteration+1)

predictions = best_model.predict(dtest)
predictions = [round(value) for value in predictions]

accuracy = accuracy_score(y_test, predictions)
print("Test accuracy:", accuracy)


# %% [markdown]
# ## Conclusion
# **In this article, we've learned how to use the XGBoost algorithm to build a classification model on the Titanic dataset in Python. We covered data loading, preprocessing, training-test data split, model creation, cross-validation, and result evaluation. Thanks to XGBoost's powerful features, you can achieve high accuracy even with more complex and larger datasets.**
# 
# **I hope this article helps you in developing your machine learning projects using the XGBoost algorithm. Happy coding!**



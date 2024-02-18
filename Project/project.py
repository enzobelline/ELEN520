
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

k = 5 #for k means

# Load the diabetes dataset
df = pd.read_csv('diabetes.csv')
print(df)

# Function to visualize the distribution of each feature
def visualize_features(df):
    for column in df.columns:
        plt.figure()
        sns.histplot(df[column], bins=20, kde=True)
        plt.title(column)
        plt.show()

# Function to perform multiple linear regression and evaluate the model
def mlRegression(df,k):
    columns_selected = "+".join(df.columns.difference(["Outcome"]))
    my_formula = "Outcome~" + columns_selected
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()
    lm_fit = smf.ols(my_formula, data=df).fit()
    print(lm_fit.summary())
    lm_fit.resid.describe()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Mean Squared Error:', mse)
    print('R-squared Score:', r2)
    # within mlRegression function
    linear_scores = cross_val_score(model, X, y, cv=len(X), scoring='neg_mean_squared_error')
    linear_loocv_error = -linear_scores.mean()
    print(f"Linear Regression LOOCV Error: {linear_loocv_error}")

    #  mlRegression kfold_error
    linear_scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
    linear_kfold_error = -linear_scores.mean()
    print(f"Linear Regression K-Fold CV Error: {linear_kfold_error}")
    if (linear_loocv_error > linear_kfold_error):
        print(f"Linear Regression K-Fold CV Error is better at {linear_kfold_error}")
    else:
        print(f"Linear Regression LOOCV Error is better at {linear_kfold_error}")

    #influence plot
    fig, ax = plt.subplots(figsize=(12,8))
    influence_plot(lm_fit, ax=ax)
    plt.title("Influence plot")
    plt.show()
    sns.pairplot(df)
    plt.show()

    #Studentized Residual vs Predicted Response:
    student_resid = OLSInfluence(lm_fit).resid_studentized_internal
    pred_vals = lm_fit.predict(X)
    plt.scatter(pred_vals, student_resid, edgecolors='k', facecolors='none')
    plt.ylabel('Studentized Residuals')
    plt.xlabel('Fitted Values')
    plt.title('Studentized Residuals vs Fitted Values')
    plt.show()

    #qqplot for residuals
    res = lm_fit.resid # residuals
    fig = sm.qqplot(res, fit=True, line='45')
    plt.title('QQ Plot of Residuals')
    plt.show()

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                            for i in range(len(X.columns))]

    print(F"This is the output of my vif_data:\n{vif_data}")
    

# Function to perform logistic regression and evaluate the model
def logisticRegression(df,k):
    # Split the dataset into features (X) and target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression()
    smLogit = sm.Logit(y,X)
    model.fit(X_train, y_train)

    result = smLogit.fit()

    print('classes:', model.classes_)
    print('coefficients:', model.coef_)
    print('intercept:', model.intercept_)
    print(result.summary())

    # Predict the target values for the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print(model)
    correlation_matrix = df.corr()

    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    # # within logistic_regression function
    # logistic_scores = cross_val_score(model, X, y, cv=len(X), scoring='neg_log_loss')
    # logistic_loocv_error = -logistic_scores.mean()
    # print(f"Logistic Regression LOOCV Error: {logistic_loocv_error}")


    # Logistic Regression  kfolderror
    logistic_scores = cross_val_score(model, X_train, y_train, cv=k, scoring='neg_log_loss')
    logistic_kfold_error = -logistic_scores.mean()
    print(f"Logistic Regression K-Fold CV Error: {logistic_kfold_error}")

    # if (logistic_loocv_error > logistic_kfold_error):
    #     print(f"Logistic Regression K-Fold CV Error is better at {logistic_kfold_error}")
    # else:
    #     print(f"Logistic Regression LOOCV Error is better at {logistic_kfold_error}")

def pcaGeneration(df):
    # The data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Standardize the features (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(data = X_pca)
    model_pca = LinearRegression()

    X_pca_reduced = X_pca_df.iloc[:, :3]  # adjust this based on the number of PCs you want to retain
    model_pca.fit(X_pca_reduced, y)

    # Print the coefficients
    print('Coefficients:', model_pca.coef_)

def rrGeneration(df):
    # The data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.logspace(-4, 4, 200)  # Feel free to adjust this

    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
    ridge_cv.fit(X_scaled, y)

    best_alpha = ridge_cv.alpha_
    print('Best alpha:', best_alpha)

    ridge = RidgeCV(alphas=[best_alpha])
    ridge.fit(X_scaled, y)
    print('Ridge coefficients:', ridge.coef_)

# Visualize the distribution of each feature
# Perform logistic regression and evaluate the model
visualize_features(df)
mlRegression(df,k)
logisticRegression(df,k)
pcaGeneration(df)
rrGeneration(df)
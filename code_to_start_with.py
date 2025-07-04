# %%
# load data
import pandas as pd

data = pd.read_csv('C:\\Users\\19647\\OneDrive\\桌面\\学习\\aiSummerCamp2025-master\\aiSummerCamp2025-master\\day1\\assignment\\data\\train.csv')
df = data.copy()
df.sample(10)
# %%
# delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df.info()
# %%
# check if there is any NaN in the dataset
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# convert categorical data into numerical data using one-hot encoding
# For example, a feature like sex with categories ['male', 'female'] would be transformed into two new binary features, sex_male and sex_female, represented by 0 and 1.
df = pd.get_dummies(df)
df.sample(10)
# %% 
# separate the features and labels
X=df.drop(columns=['Survived'])
y=df['Survived']
# %%
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# build model
# build three classification models
# SVM, KNN, Random Forest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
svm_model = SVC(random_state=42)
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier(random_state=42)
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
# %%
# predict and evaluate
from sklearn.metrics import accuracy_score, classification_report
def evaluate_model(model, X_test, y_test):
 y_pred = model.predict(X_test)
 accuracy = accuracy_score(y_test, y_pred)
 report = classification_report(y_test, y_pred)
 return accuracy, report
# %%
# Evaluate SVM
svm_accuracy, svm_report = evaluate_model(svm_model, X_test, y_test)
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:\n", svm_report)
# %%
# Evaluate KNN
knn_accuracy, knn_report = evaluate_model(knn_model, X_test, y_test)
print("KNN Accuracy:", knn_accuracy)
print("KNN Classification Report:\n", knn_report)
# %%
# Evaluate Random Forest
rf_accuracy, rf_report = evaluate_model(rf_model, X_test, y_test)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:\n", rf_report)
# %%

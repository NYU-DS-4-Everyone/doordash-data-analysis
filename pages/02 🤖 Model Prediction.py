import streamlit as st
import pandas as pd
from PIL import Image
import sklearn.metrics as sk_metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from codecarbon import EmissionsTracker

url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
st.image(url,  output_format="PNG", width=300)

st.title("Model Prediction")

df_unclean = pd.read_csv("ifood-data.csv")
df = df_unclean.dropna()
df = df[df["Year_Birth"] > 1940]
df['Education'] = df['Education'].astype('category').cat.codes
df['Marital_Status'] = df['Marital_Status'].astype('category').cat.codes
df = df.drop(["Dt_Customer"], axis = 1)
df = df.drop(["ID"], axis = 1)

X = df.drop(labels = ['Response'], axis = 1)
y = df["Response"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

st.multiselect("Select Parameters", df.columns)

tracker = EmissionsTracker()
tracker.start()

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
log_results = logmodel.predict(X_test)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
tree_results = clf.predict(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_results = knn.predict(X_test)

emissions = tracker.stop()
print(f"Estimated emissions for training the model: {emissions:.4f} kg of CO2")

st.metric(label = "Log Accuracy", value = round(metrics.accuracy_score(y_test, log_results)*100, 2))
st.metric(label = "Tree Accuracy", value = round(metrics.accuracy_score(y_test, tree_results)*100, 2))
st.metric(label = "kNN Accuracy", value = round(metrics.accuracy_score(y_test, knn_results)*100, 2))

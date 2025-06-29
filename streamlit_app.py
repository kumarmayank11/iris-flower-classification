
import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

st.title("Iris Flower Classification")
sepal_length = st.slider("Sepal length", float(X[:,0].min()), float(X[:,0].max()))
sepal_width = st.slider("Sepal width", float(X[:,1].min()), float(X[:,1].max()))
petal_length = st.slider("Petal length", float(X[:,2].min()), float(X[:,2].max()))
petal_width = st.slider("Petal width", float(X[:,3].min()), float(X[:,3].max()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
st.write("Prediction:", iris.target_names[prediction][0])

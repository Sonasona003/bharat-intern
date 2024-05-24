import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

data = load_iris()
X = data.data  
model = LogisticRegression(max_iter=1000)

sepal_length = float(input("Enter sepal length (in cm): "))
sepal_width = float(input("Enter sepal width (in cm): "))
petal_length = float(input("Enter petal length (in cm): "))
petal_width = float(input("Enter petal width (in cm): "))

input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

model.fit(X, y)

predicted_species = model.predict(input_features)

species_labels = ['Setosa', 'Versicolor', 'Virginica']
predicted_species_label = species_labels[predicted_species[0]]
print(f"The predicted species is: {predicted_species_label}")
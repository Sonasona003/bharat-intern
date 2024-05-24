import numpy as np
from sklearn.linear_model import LinearRegression

model = LinearRegression()

feature_names = ['bedrooms', 'bathrooms', 'area_sqft', 'garage', 'year_built']
features = []

for feature_name in feature_names:
    if feature_name == 'area_sqft':
        feature_value = float(input("Enter the area in square feet: "))
    else:
        feature_value = float(input(f"Enter the number of {feature_name}: "))
    features.append(feature_value)
area_sqft = features[2]
area_sqm = area_sqft * 0.092903
features[2] = area_sqm

features = np.array(features).reshape(1, -1)


X_train = np.array([[3, 2, 167.22, 2, 2005],  # 1800 sqft converted to sqm
                    [4, 3, 204.38, 2, 1998],  # 2200 sqft converted to sqm
                    [2, 1, 111.48, 1, 2010]])  # 1200 sqft converted to sqm
y_train = np.array([300000, 400000, 250000])

model.fit(X_train, y_train)

predicted_price = model.predict(features)

predicted_price = abs(predicted_price)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
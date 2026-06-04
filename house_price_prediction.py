import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Read the dataset
housing = pd.read_csv("Housing.csv")

# Display basic information
print("Total Rows and Columns:", housing.shape)
print("\nSample Data:")
print(housing.head())

# Define input features and target variable
features = housing.loc[:, ["area", "bedrooms", "bathrooms"]]
target = housing["price"]

# Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42
)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Display training information
print("\nModel Trained Successfully!")

print("Intercept:", round(lr_model.intercept_, 2))
print("Coefficients:")

for feature, coef in zip(features.columns, lr_model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nHouse Price Prediction Model Ready!")

# Get user input
print("\nEnter House Details")
house_area = float(input("Area (sq.ft): "))
house_bedrooms = int(input("Bedrooms: "))
house_bathrooms = int(input("Bathrooms: "))

# Create dataframe for prediction
input_data = pd.DataFrame(
    [[house_area, house_bedrooms, house_bathrooms]],
    columns=["area", "bedrooms", "bathrooms"]
)

# Predict house price
price_prediction = lr_model.predict(input_data)

print("\nEstimated House Price: ₹{:,.0f}".format(price_prediction[0]))
print("Approximate Value: {:.2f} Lakhs".format(price_prediction[0] / 100000))

# Predict on test data
test_predictions = lr_model.predict(X_test)

# Performance metrics
r2 = r2_score(y_test, test_predictions)
mae = mean_absolute_error(y_test, test_predictions)

print("\nModel Evaluation")
print("R² Score =", round(r2, 4))
print("Mean Absolute Error =", round(mae, 2))

# Visualization
plt.figure(figsize=(12, 5))

# Actual vs Predicted Line Graph
plt.subplot(1, 2, 1)
plt.plot(
    range(20),
    y_test.iloc[:20],
    marker='o',
    label='Actual Price'
)

plt.plot(
    range(20),
    test_predictions[:20],
    marker='x',
    label='Predicted Price'
)

plt.title("Actual vs Predicted Prices")
plt.xlabel("Sample Houses")
plt.ylabel("Price")
plt.legend()

# Scatter Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, test_predictions)

plt.title("Prediction Accuracy")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.tight_layout()
plt.show()
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([52, 55, 61, 70, 82])

lin = LinearRegression()
lin.fit(X, y)

y_lin = lin.predict(X)
x6 = np.array([[6]])
lin_6 = lin.predict(x6)
mse_lin = mean_squared_error(y, y_lin)

print("Linear Model")
print("y =", round(lin.intercept_, 2), "+", round(lin.coef_[0], 2), "x")
print("x=6:", lin_6[0])
print("MSE:", mse_lin)

poly = PolynomialFeatures(4)
X_p = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_p, y)

y_poly = poly_reg.predict(X_p)
poly_6 = poly_reg.predict(poly.transform(x6))
mse_poly = mean_squared_error(y, y_poly)

print("\nPolynomial Model (Degree 4)")
print("x=6:", poly_6[0])
print("MSE:", mse_poly)

print("\nBias–Variance")
print("Linear: High Bias, Low Variance")
print("Polynomial: Low Bias, High Variance")

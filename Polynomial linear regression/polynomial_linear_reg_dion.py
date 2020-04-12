#polynomial regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values#when we add only [:,1] => we get a vector to solve issue we use [:,1:2](same thing but rep as a matrix now)
y = dataset.iloc[:, 2:3].values

# =============================================================================
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# =============================================================================

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#Linear regression model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)#to get the polynomial matrix of x(x^2 and so on)
lin_reg2 = LinearRegression() #to fit poly matrix x in model(since were doing linear polynomial regression)
lin_reg2.fit(X_poly,y)

#visualising the linear regression model
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show()

#visualising the linear polynomial model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1)) 
#the above code increases the resolution of the plotted graph by decreasing the step side to 0.1
plt.scatter(X,y,color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show()

#predicting a new result with linear regression
lin_reg.predict([[6.5]])

#predicting a new result with linear polynomial regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
















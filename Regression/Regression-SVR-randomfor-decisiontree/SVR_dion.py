#SVR
#NOTE=> SVR doesnt apply feature scaling automatically

#regression template

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)




#create your regressor


#predicting a new result with polynomial regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualising the regression results
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bluff (SVR)')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show()


#visualising the regression results (for higher resultion and smooter curve)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bluff (SVR)')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show()




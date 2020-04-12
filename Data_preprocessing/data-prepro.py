#data preprosessing

#import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd#for import and managemnet of datasets

#import the datasets
dataset = pd.read_csv('Data1.csv')
x = dataset.iloc[:,:-1].values # we will take all the rows,and all the columns except the last one(-1)
y = dataset.iloc[:,3].values #we will take all rows and last column


#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0) #ctr +i to see docs
imputer = imputer.fit(x[:,1:3])#choose columns that have missing data
x[:,1:3] = imputer.transform(x[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
#above  we finsihed dummy encoding the countries
#to encode the purchased column
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#splitting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state = 0)#0.2=> 20% in test and 80% in training set

# =============================================================================
# #feature scaling
# from sklearn.preprocessing import StandardScaler
# sc_x= StandardScaler()
# x_train = sc_x.fit_transform(x_train)
# x_test = sc_x.transform(x_test)#dont need to fit for test set since already fitted for training set
# =============================================================================

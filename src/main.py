
import pandas as pd
import sys
from impyute.imputation.cs import fast_knn
import numpy as np



#data: https://www.kaggle.com/thebasss/currency-exchange-rates

data = pd.read_csv("data/currency_exchange_rates_02-01-1995_-_02-05-2018.csv")



sys.setrecursionlimit(100000) #Increase the recursion limit of the OS

cp = data.copy()
cp = cp.drop('Date', 1)
print(np.count_nonzero(cp.values))
print(np.count_nonzero(~np.isnan(cp.values)))

"""
Imputation Using k-NN

The k nearest neighbours is an algorithm that is used for simple classification. 
The algorithm uses ‘feature similarity’ to predict the values of any new data points. 
This means that the new point is assigned a value based on how closely it resembles the points in the training set. 
This can be very useful in making predictions about the missing values by finding the k’s closest neighbours to the 
observation with missing data and then imputing them based on the non-missing values in the neighbourhood. 
"""

# start the KNN training
imputed_training=fast_knn(cp.values, k=30).tolist()
#print(np.count_nonzero(~np.isnan(imputed_training)))

#return to pandas
index = data.columns.values.tolist()
df = pd.DataFrame(imputed_training)
df.columns = index[1:]
df.insert(loc=0, column='Date', value=data['Date'].values)
print(df)

#use arima
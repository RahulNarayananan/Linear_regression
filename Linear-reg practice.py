#importing necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

#reading the data file and extracting values from csv file
read=pd.read_csv('Foodtruck.csv')
x=read['Population'].values
y=read['Profit'].values

#reshaping array and fitting into model to check accuracy of the model
x=x.reshape(-1,1)
model= LinearRegression()
model.fit(x,y)
accuracy=model.score(x,y) 

#finding slope and constant terms of eq y=mx+c
const=model.intercept_
slope=model.coef_

#plotting the various values and the regression line
plt.scatter(x, y, color = 'blue', label='points')
plt.title('Linear Regression')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.plot(x,model.predict(x),color='red',label='regression line')
plt.legend(loc=4)
plt.show()

#taking input to check expected profit with respect to the population of that place
pop=float(input('enter population: '))
pop=np.array(pop).reshape(-1,1)
y_pred=model.predict(pop)
print('the expected profit is : ',y_pred[0])

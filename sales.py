import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load the data from csv file to Pandas DataFrame

data=pd.read_csv('advertising.csv')

# printing the first 5 rows of the dataframe
data.head()
# printing the last 5 rows of the dataframe
data.tail()

# number of rows and Columns
data.shape

# getting some informations about the data

data.info()

# check the number of missing values in each column
data.isnull().sum()

# show seaborn relationship
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr())
plt.show()

#drop a column
x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])

#training data 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

data = pd.DataFrame(data={"Predicted Sales": ypred.flatten()})
print(data)

model.fit(xtest, ytest)
ypred1 = model.predict(xtrain)

data = pd.DataFrame(data={"Predicted Sales": ypred1.flatten()})
print(data)
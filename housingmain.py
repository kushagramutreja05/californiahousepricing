#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pickle
#fetch the dataset
df=fetch_california_housing()
#inspect the dataset
print(df.keys())
print(df.DESCR)
print(df.data)
print(df.target)
print(df.feature_names)

#create a dataframe
dataset=pd.DataFrame(df.data,columns=df.feature_names)
print(dataset)
print(dataset.describe())
#add the target variable dependent to the dataframe
dataset['price'] = df.target
print(dataset.head())
x = dataset[df.feature_names]
y=dataset['price']
print(dataset.isnull())
print(dataset.corr())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model = LinearRegression()
model.fit(x_train, y_train)
print(model.coef_)
print(model.intercept_)
print(model.get_params())
y_pred = model.predict(x_test)
plt.scatter(y_test, y_pred)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('True Prices vs Predicted Prices')
plt.show()
residual =y_test-y_pred
sns.displot(residual,kind="kde")
plt.show()
plt.scatter(y_pred,residual)
plt.show()
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))
score=r2_score(y_test,y_pred)
print(score)
df.data[0].reshape(1,-1)
new_data=scaler.transform(df.data[1].reshape(1,-1))
print (model.predict(new_data))

pickle.dump(model,open('lmodel.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))




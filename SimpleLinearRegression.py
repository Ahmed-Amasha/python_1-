import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_set = pd.read_csv("Salary_Data.csv")
X= data_set.iloc[: , : -1].values
Y= data_set.iloc[: , : 1].values

x_train, x_test, y_train, y_test= train_test_split(X,Y, test_size=0.2)
L_R = LinearRegression()
L_R.fit(x_train, y_train)
y_pred = L_R.predict(x_test)
y_pred_train= L_R.predict(x_train)

plt.scatter(x_train, y_train, c= "g")
plt.plot(x_train, L_R.predict(x_train), c="r")
plt.title("Years of experiance _ Training Data")
plt.xlabel("experiance")
plt.ylabel("Salary")
plt.show()

plt.scatter(x_test, y_test, c= "g")
plt.plot(x_test, y_test, c="r")
plt.title(" Years of experiance _Test Data")
plt.xlabel("experiance")
plt.ylabel("Salary")
plt.show()

#Z = evaluation 
#Z.mean_absolute_error(Y, y_pred_train)
git remote add origin https://github.com/Ahmed-Amasha/My-data-analysis-code.git
git branch -M main
git push -u origin main

from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


boston_market_data = load_boston()
boston_market_data_p = pd.DataFrame(boston_market_data.data, columns=[boston_market_data.feature_names])
boston_market_data_p.head()

print(boston_market_data_p.describe())

scaler = StandardScaler()
boston_market_data['data'] = scaler.fit_transform(boston_market_data['data'])

boston_market_data_p = pd.DataFrame(boston_market_data.data, columns=[boston_market_data.feature_names])
boston_market_data_p.head()

print(boston_market_data_p.describe())

boston_train_data, boston_test_data, \
boston_train_target, boston_test_target = \
train_test_split(boston_market_data['data'],boston_market_data['target'], test_size=0.1)

print("--- Training dataset:")
print("boston_train_data:", boston_train_data.shape)
print("boston_train_target:", boston_train_target.shape)

print("--- Testing dataset:")
print("boston_test_data:", boston_test_data.shape)
print("boston_test_target:", boston_test_target.shape)


linear_regression = LinearRegression()
linear_regression.fit(boston_train_data, boston_train_target)

id=5
linear_regression_prediction = linear_regression.predict(boston_test_data[id,:].reshape(1,-1))

print("--- Check for ID=5:")
print(boston_test_data[id,:].shape)
print(boston_test_data[id,:].reshape(1,-1).shape)

print("--- Score:")
print("Mean squared error of a learned model: %.2f" %
mean_squared_error(boston_test_target, linear_regression.predict(boston_test_data)))

print('Variance score: %.2f' % r2_score(boston_test_target, linear_regression.predict(boston_test_data)))

scores = cross_val_score(LinearRegression(), boston_market_data['data'], boston_market_data['target'], cv=4)
print('Cross-validation')
print(scores)
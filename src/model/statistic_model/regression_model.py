from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb


def linear_model(train_data, train_target, test_data, test_target):
    clf = LinearRegression()
    clf.fit(train_data, train_target)
    # test_pred = clf.predict(test_data)
    score = mean_squared_error(test_target, clf.predict(test_data))
    print("KNeighborsRegressor: ", score)


def knn_model(train_data, train_target, test_data, test_target):
    clf = KNeighborsRegressor(n_neighbors=3)
    clf.fit(train_data, train_target)
    # test_pred = clf.predict(test_data)
    score = mean_squared_error(test_target, clf.predict(test_data))
    print("KNeighborsRegressor: ", score)


def decision_tree_model(train_data, train_target, test_data, test_target):
    clf = DecisionTreeRegressor()
    clf.fit(train_data, train_target)
    test_pred = clf.predict(test_data)
    score = mean_squared_error(test_target, test_pred)
    print("DecisionTreeRegressor: ", score)


def random_forest_model(train_data, train_target, test_data, test_target):
    # 200棵树模型
    clf = RandomForestRegressor(n_estimators=200)
    clf.fit(train_data, train_target)
    test_pred = clf.predict(test_data)
    score = mean_squared_error(test_target, test_pred)
    print("DecisionTreeRegressor: ", score)


def lgb_model(train_data, train_target, test_data, test_target):
    clf = lgb.LGBMRegressor(
        learning_rate=0.01,
        max_depth=-1,
        n_estimators=5000,
        boosting_type='gbdt',
        random_state=2019,
        objective='regression'
    )
    clf.fit(X=train_data, y=train_target, eval_metric='MSE', verbose=50)
    score = mean_squared_error(test_target, clf.predict(test_data))
    print("lightGbm:  ", score)

from sklearn.model_selection import GridSearchCV, train_test_split,RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def grid_search(model, parameters, train, target):
    train_data, train_target, test_data, test_target = train_test_split(train, target, test_size=2.0, random_state=0)
    model.fit(train_data, train_target)
    clf = GridSearchCV(model, parameters, cv=5)
    clf.fit(train_data, train_target)
    score_test = mean_squared_error(test_target, clf.predict(test_data))
    print("model gridSearch test MSE: ", score_test)
    print("输出训练时间和验证指标的信息")
    print(sorted(clf.cv_results_.keys()))


def rand_search(model, parameters, train, target):
    train_data, train_target, test_data, test_target = train_test_split(train, target, test_size=2.0, random_state=0)
    model.fit(train_data, train_target)
    clf = RandomizedSearchCV(model, parameters, cv=5)
    clf.fit(train_data, train_target)
    score_test = mean_squared_error(test_target, clf.predict(test_data))
    print("model gridSearch test MSE: ", score_test)
    print("输出训练时间和验证指标的信息")
    print(sorted(clf.cv_results_.keys()))


if __name__ == "__main__":
    train_data = pd.read_csv()
    train_target = pd.read_csv()
    parameters = {'n_estimators': [50, 100, 200], 'max_depth': [1, 2, 3]}
    grid_search(RandomForestRegressor(), parameters, train_data, train_target)

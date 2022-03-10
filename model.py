import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def train():
    df = pd.read_csv("data.csv")
    df = df.drop(["date", "street", "city", "statezip", "country"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["price"], axis=1), df["price"], test_size=0.33
    )
    print(X_train)
    mdl = LinearRegression()
    mdl.fit(X_train.values, y_train)
    y_pred = mdl.predict(X_test.values)
    print("Mean absolute error : ", mean_absolute_error(y_test, y_pred))
    pickle.dump(mdl, open("saved_model.pkl", "wb"))


def predict(
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition,
    sqft_above,
    sqft_basement,
    yr_built,
    yr_renovated,
):
    params = locals()
    x = pd.DataFrame(data=[params.values()], columns=params.keys())
    mdl = pickle.load(open("saved_model.pkl", "rb"))
    price = mdl.predict(x)
    return price[0]


if __name__ == "__main__":
    # exemple
    res = predict(3.0, 1.5, 1340, 7912, 1.5, 0, 0, 3, 1340, 0, 1955, 2005)
    print(res)

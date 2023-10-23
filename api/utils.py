import pathlib
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder

def get_data():
    DATA_DIR = pathlib.Path.cwd().parent / 'data'
    clean_data_path = DATA_DIR / 'processed' / 'ames_clean.pkl'

    data = pd.read_pickle(clean_data_path)
    raw = data.copy()
    raw = data.drop(["SalePrice"], axis=1)
    data = feature_engineering(data)

    X = data.drop(["SalePrice"], axis=1)
    y = data["SalePrice"]
    return raw, X, y

def get_model():
    return Ridge(alpha=10)

def check_fields(json, fields):
    for field in json:
        if field not in fields:
            return False
    return True

def feature_engineering(data):
    categorical_columns = []
    ordinal_columns = []
    for col in data.select_dtypes('category').columns:
        if data[col].cat.ordered:
            ordinal_columns.append(col)
        else:
            categorical_columns.append(col)

    ordinal_encoder = OrdinalEncoder()
    data[ordinal_columns] = ordinal_encoder.fit_transform(data[ordinal_columns])
    data[categorical_columns] = data[categorical_columns].astype(str)

    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True, prefix=categorical_columns)

    data["SqFtPerRoom"] = data["Gr.Liv.Area"] / (data["TotRms.AbvGrd"] +
                                                data["Full.Bath"] +
                                                data["Half.Bath"] +
                                                data["Kitchen.AbvGr"])

    data['Total_Home_Quality'] = data['Overall.Qual'] + data['Overall.Cond']

    data['Total_Bathrooms'] = (data['Full.Bath'] + (0.5 * data['Half.Bath']) +
                               data['Bsmt.Full.Bath'] + (0.5 * data['Bsmt.Half.Bath']))

    data["HighQualSF"] = (data["Gr.Liv.Area"] + data["X1st.Flr.SF"] + data["X2nd.Flr.SF"] +
                          0.5 * data["Garage.Area"] + 0.5 * data["Total.Bsmt.SF"] +
                          data["Mas.Vnr.Area"])
    return data

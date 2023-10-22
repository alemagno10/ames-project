import pathlib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def get_data():
    DATA_DIR = pathlib.Path.cwd().parent / 'data'
    clean_data_path = DATA_DIR / 'processed' / 'ames_clean.pkl'
    print(DATA_DIR)

    data = pd.read_pickle(clean_data_path)
    X = data.drop(["SalePrice"], axis=1)
    y = data["SalePrice"]

    return X, feature_engineering(X), y

def get_model():
    return LinearRegression()

def check_fields(json, fields):
    # if len(json.c) != len(fields):
    #     return False

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

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first'), categorical_columns),
        ],
        remainder='passthrough',
    )

    preprocessor.fit(data)
    array_transformed = preprocessor.transform(data)

    new_categorical_columns = preprocessor.named_transformers_['onehot'].get_feature_names_out()
    new_columns = new_categorical_columns.tolist() + [col for col in data.columns if col not in categorical_columns]

    data = pd.DataFrame(array_transformed, columns=new_columns)
    
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

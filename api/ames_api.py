from flask import Flask, request, jsonify
from utils import *

app = Flask(__name__)

raw, X, y = get_data()
model = get_model()
model.fit(X, y)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api', methods=['GET'])
def teste():
    return raw.iloc[10].to_dict(), 200

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json()
    raw, X, y = get_data()
    df = pd.DataFrame([data])
    df = df.T

    # if not check_fields(data, list(X.columns)):
    #     return data, 200
        # return jsonify({'error: Verifique os campos enviados'}), 400
    
    # predicted_y = model.predict(feature_engineering(df.iloc[0]))
    # print(predicted_y)

    # response = {
    #     'message': 'Regressão concluída com sucesso',
    #     'predicted_y': predicted_y.tolist()
    # }

    return df.to_dict(), 200
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)
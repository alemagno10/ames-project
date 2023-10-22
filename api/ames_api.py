from flask import Flask, request, jsonify
from utils import *
import random

app = Flask(__name__)

raw, X, y = get_data()
model = get_model()
model.fit(X, y)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api', methods=['GET'])
def teste():
    i = random.randint(0,2877)
    return raw.iloc[i].to_dict(), 200

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    if not check_fields(data.keys(), list(raw.columns)):
        return jsonify({'error: Verifique os campos enviados'}), 400
    
    predicted_y = model.predict(feature_engineering(df))

    response = {
        'message': 'Regressão concluída com sucesso',
        'predicted_y': predicted_y.tolist()
    }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)
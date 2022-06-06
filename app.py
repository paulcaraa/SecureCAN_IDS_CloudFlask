from flask import Flask, request, jsonify
import numpy as np
import pickle
knn_basic_model = pickle.load(open('knn_basic.pkl','rb'))
knn_replay_model = pickle.load(open('knn_replay.pkl','rb'))
knn_universal_model = pickle.load(open('knn_universal.pkl','rb'))

lr_basic_model = pickle.load(open('lr_basic.pkl','rb'))
lr_replay_model = pickle.load(open('lr_replay.pkl','rb'))
lr_universal_model = pickle.load(open('lr_universal.pkl','rb'))

dt_basic_model = pickle.load(open('dt_basic.pkl','rb'))
dt_replay_model = pickle.load(open('dt_replay.pkl','rb'))
dt_universal_model = pickle.load(open('dt_universal.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return "Paul Ioan Carabas Diploma Project"
@app.route('/predict',methods=['POST'])
def predict():
    model_type = request.form.get('model')
    dt = request.form.get('dt')
    id = request.form.get('id')
    b0 = request.form.get('b0')
    b1 = request.form.get('b1')
    b2 = request.form.get('b2')
    b3 = request.form.get('b3')
    b4 = request.form.get('b4')
    b5 = request.form.get('b5')
    b6 = request.form.get('b6')
    b7 = request.form.get('b7')

    input_query = np.array([[dt, id, b0, b1, b2, b3, b4, b5, b6, b7]], dtype=float)

    model_type = str(model_type)

    if model_type == "knn_basic":
        result = knn_basic_model.predict(input_query)[0]
    elif model_type == "knn_replay":
        result = knn_replay_model.predict(input_query)[0]
    elif model_type == "knn_universal":
        result = knn_replay_model.predict(input_query)[0]
    elif model_type == "lr_basic":
        result = lr_basic_model.predict(input_query)[0]
    elif model_type == "lr_replay":
        result = lr_replay_model.predict(input_query)[0]
    elif model_type == "lr_universal":
        result = lr_replay_model.predict(input_query)[0]
    elif model_type == "dt_basic":
        result = dt_basic_model.predict(input_query)[0]
    elif model_type == "dt_replay":
        result = dt_replay_model.predict(input_query)[0]
    elif model_type == "dt_universal":
        result = dt_replay_model.predict(input_query)[0]

    return jsonify({'intrusion':str(result)})
if __name__ == '__main__':
    app.run(debug=True)
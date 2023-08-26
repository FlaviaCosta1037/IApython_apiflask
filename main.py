from flask import Flask, request, jsonify
import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

app = Flask(__name__)

params = {
    'dbname': 'blazedouble',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': 5432
}

clf = DecisionTreeClassifier()
le_cor = LabelEncoder()

def hora_to_decimal(hora_obj):
    return hora_obj.hour + hora_obj.minute/60

def inicializar_modelo():
    global clf, le_cor

    conn = psycopg2.connect(**params)
    query = "SELECT cor, data, hora FROM resultados;"
    data = pd.read_sql(query, conn)
    conn.close()

    data['hora_decimal'] = data['hora'].apply(hora_to_decimal)
    data['cor_encoded'] = le_cor.fit_transform(data['cor'])

    X_train, X_test, y_train, y_test = train_test_split(data['hora_decimal'], data['cor_encoded'], test_size=0.2)
    clf.fit(X_train.values.reshape(-1, 1), y_train)

    # Garantir que 'branca' esteja no LabelEncoder, mesmo que não esteja nos dados
    cores_possiveis = data['cor'].unique().tolist() + ['branca']
    le_cor.fit(cores_possiveis)

def prever_cor(hora_string):
    hora_obj = datetime.strptime(hora_string, "%H:%M").time()
    hora_decimal = hora_to_decimal(hora_obj)
    predicao_encoded = clf.predict([[hora_decimal]])
    predicao = le_cor.inverse_transform(predicao_encoded)
    return predicao[0]

def prever_cor_com_probabilidade(hora_string):
    hora_obj = datetime.strptime(hora_string, "%H:%M").time()
    hora_decimal = hora_to_decimal(hora_obj)
    predicao_encoded = clf.predict([[hora_decimal]])
    probabilidade = clf.predict_proba([[hora_decimal]])
    predicao = le_cor.inverse_transform(predicao_encoded)
    prob_branca = probabilidade[0][le_cor.transform(['branca'])[0]]
    return predicao[0], prob_branca

@app.route('/prever_cor', methods=['GET'])
def api_prever_cor():
    hora_input = request.args.get('hora')
    if not hora_input:
        return jsonify(error="Hora não fornecida"), 400
    cor_predita = prever_cor(hora_input)
    return jsonify(cor=cor_predita)

@app.route('/prever_cor_com_probabilidade', methods=['GET'])
def api_prever_cor_com_probabilidade():
    hora_input = request.args.get('hora')
    if not hora_input:
        return jsonify(error="Hora não fornecida"), 400
    cor_predita, prob_branca = prever_cor_com_probabilidade(hora_input)
    return jsonify(cor=cor_predita, probabilidade_branca=f"{prob_branca:.2%}")

if __name__ == '__main__':
    inicializar_modelo()
    app.run(debug=True)

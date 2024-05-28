#meu primeiro código com rede neural artificial

from sklearn.neural_network import MLPClassifier

import pickle

from sklearn.metrics import accuracy_score, classification_report

with open('credit.pkl','rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_test, y_credit_test = pickle.load(f)

'''
número de camadas ocultas:

(3+1)/2 = 2

'''

rede_neural_credit = MLPClassifier(max_iter= 1500, verbose= True, tol= 0.00001,
                                   solver= 'adam', activation='relu',
                                   hidden_layer_sizes= (10,10))

rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = rede_neural_credit.predict(X_credit_test)

precisao = accuracy_score(y_credit_test, previsoes)

print(f'A precisão deste algoritmo é de {precisao*100} %')

print(classification_report(y_credit_test, previsoes))
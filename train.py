"""# 1 - Importando os módulos necessários"""

import os
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""# Definindo funções adicionais"""

import os
import random
import numpy as np
import random as python_random
 
def reset_seeds():
   os.environ['PYTHONHASHSEED']=str(42)
   tf.random.set_seed(42)
   np.random.seed(42)
   random.seed(42)

"""# 2 - Fazendo a leitura do dataset e atribuindo às respectivas variáveis"""

data = pd.read_csv('https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')

"""# Dando uma leve olhada nos dados"""

data.head()

"""# 3 - Preparando o dado antes de iniciar o treino do modelo"""

X=data.drop(["fetal_health"], axis=1)
y=data["fetal_health"]

columns_names = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns_names)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

y_train = y_train -1
y_test = y_test - 1

"""# 4 - Criando o modelo e adicionando as camadas"""

reset_seeds()
model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1], )))
model.add(Dense(10, activation='relu' ))
model.add(Dense(10, activation='relu' ))
model.add(Dense(3, activation='softmax' ))

"""# 5 - Compilando o modelo

"""

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

import mlflow

MLFLOW_TRACKING_URI = 'https://dagshub.com/renansantosmendes/mlops_puc_230523.mlflow'
MLFLOW_TRACKING_USERNAME = 'renansantosmendes'
MLFLOW_TRACKING_PASSWORD = 'cc41cc48f8e489dd5b87404dd6f9720944e32e9b'
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.tensorflow.autolog(log_models=True, 
                          log_input_examples=True,
                          log_model_signatures=True)

"""# 6 - Executando o treino do modelo"""

with mlflow.start_run(run_name='experiment_01') as run:
  model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=3)

# from email import header
# from fileinput import filename
# from importlib.resources import path
# import json
# import os
# from unittest.mock import patch
# import scikeras
# import keras_tuner
# import numpy as np
# import tensorflow as tf
# import keras_tuner as kt
# import pandas as pd
# from keras.callbacks import EarlyStopping
# from keras_tuner.tuners import BayesianOptimization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import train_test_split
# from imblearn.combine import SMOTEENN
# from functools import partial
# from flask import Flask, jsonify, render_template, request, send_from_directory
import json
import os
import scikeras
import keras_tuner
import pandas as pd
import numpy as np
import pickle
import shutil
from joblib import dump
from functools import partial
from flask import Flask, jsonify, render_template, request, send_file
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.metrics import confusion_matrix

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = './static/upload/'
app.config['RESULT_SUMMARY'] = 'result/bayesian/'
app.config['RESULT_BEST_HPS'] = 'result/hyperparameter/'
app.config['RESULT_MODEL'] = 'result/model/'


def CreateDirectory(path):
    dir = path
    if os.path.exists(dir):
        shutil.rmtree(dir)

    os.makedirs(dir)


# route untuk menampilkan halaman awal (index.html) -> HOME
@app.route('/')
def home():
    CreateDirectory('result')
    CreateDirectory('result/hyperparameter')
    CreateDirectory('result/model')
    return render_template('index.html')

# route untuk upload dataset
@app.route('/tuning', methods=['GET', 'POST'])
def tuning():
    if request.method == 'POST':
        if request.files:
            fileDataset = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fileDataset.filename)
            fileDataset.save(file_path)
            return render_template('tuning.html')

    return render_template('tuning.html')


# route untuk mengecek dataset
@app.route('/tuning/check', methods=['POST'])
# Fungsi tentang informasi dataset
def CheckDataset():
    if request.method == 'POST':
        getData = request.get_data()
        filename = getData.decode('utf-8')
        path = app.config['UPLOAD_FOLDER'] + filename

        dataset = pd.read_csv(path)

        # Check if dataset contain string
        IsAnyString = not any(dataset.applymap(np.isreal).all())
        
        # check if any missing value in dataset
        dataKosong = int(dataset.isna().sum().sum())

        # message if dataset not acceptable to tune
        message = []

        # if no missing value instead and all data is numeric type, then balancing data
        if dataKosong == 0 and IsAnyString == False:
            dataset = BalancingData(dataset, filename)
            status = 1 # memenuhi syarat
        else:
            dataset = dataset
            status = 0 # tidak memenuhi syarat
            
            if dataKosong > 0:
                message.append('There is missing value detected in your dataset.')
            
            if IsAnyString== False:
                message.append('Your dataset contain string value (not column name).')


        jumlahData = len(dataset.index)
        
        result = {
            'nama-dataset': filename,
            'file-path': path,
            'jumlah-data': jumlahData,
            'jumlah-atribut':  GetInfo(dataset)[1],
            'target': GetInfo(dataset)[0],
            'data-kosong': dataKosong,
            'status': status,
            'message': message
        }
            
        return jsonify(result)
    else:
        pesan= "Tidak ada method POST"
        return pesan


# Fungsi Tuning
@app.route('/tuning/process', methods=['POST'])
def Tuning():
    if request.method == 'POST':
        getData = request.get_data()
        filePath = getData.decode('utf-8')
        dataset = pd.read_csv(filePath)

        tuner = Tuner(dataset)[0]
        max_trials = Tuner(dataset)[1]
        earlystopper = EarlyStopper()
        
        # split data
        X_train = SplitData(dataset)['X_train']
        y_train = SplitData(dataset)['y_train']

        # proses tuning
        tuner.search(X_train, y_train, epochs= 300, validation_split= 0.2, callbacks = [earlystopper])

        # print hyperparameter paling optimal
        best_hps= tuner.get_best_hyperparameters(num_trials= 1)[0]

        # save best hyperparameter
        pickle.dump(best_hps, open(app.config['RESULT_BEST_HPS'] + 'hyperparameter.pkl', "wb"))

        best_hps_values = best_hps.values
        best_hps_values['unit_input'] = GetInfo(dataset)[1]
        best_hps_values['accuracy'] = TunerSearchSummary(max_trials - 1)[0]
        best_hps_values['val_accuracy'] = TunerSearchSummary(max_trials - 1)[1]

        return jsonify(best_hps_values)


# Fungsi untuk membuat model
@app.route('/build', methods=['POST'])
def BuildModel():
    if request.method == 'POST':
        getData = request.get_data()
        filePath = getData.decode('utf-8')
        dataset = pd.read_csv(filePath)

        tuner = Tuner(dataset)[0]
        earlystopper = EarlyStopper()
        best_hps = pickle.load(open(app.config['RESULT_BEST_HPS'] + 'hyperparameter.pkl', "rb"))

        # split data
        X_train = SplitData(dataset)['X_train']
        y_train = SplitData(dataset)['y_train']

        # fit model menggunakan hasil tuning hyperparameter dan melatihnya
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(X_train, y_train, epochs= 500, validation_split= 0.2, callbacks= [earlystopper])
        
        # save model
        model.save(app.config['RESULT_MODEL'] + "model.h5")

        # hasil evaluasi
        hasilEvaluasi = Evaluasi(model, dataset, history)

        return hasilEvaluasi


@app.route('/download')
def download():
    return send_file(app.config['RESULT_MODEL'] + 'model.h5', as_attachment=True)


# Fungsi untuk mengetahui nama kelas (target) dan jumlah atribut
def GetInfo(dataset):
    jumlahAtribut = len(dataset.columns) - 1
    namaKelas = dataset.columns[jumlahAtribut]

    return [namaKelas, jumlahAtribut]


# Fungsi untuk sampling data -> menangani data tidak seimbang
def SamplingData(dataset):
    sm =  SMOTEENN(random_state= 42)
    
    target = GetInfo(dataset)[0]

    cols = dataset.columns.tolist()
    cols = [c for c in cols if c not in [target]]

    # X: data atribut dan Y: data kelas/target
    X = dataset[cols]
    Y = dataset[target]

    # proses sampling dengan SMOTE
    dataKosong = int(dataset.isna().sum().sum())

    if dataKosong == 0:
        X_smote, Y_smote = sm.fit_resample(X, Y)
        X_smote[target] = Y_smote
        dataset = X_smote
    
    return dataset

# Fungsi mengecek apakah dataset merupakan kasus binary classification atau bukan
def BalancingData(dataset, filename):
    # class target
    target = GetInfo(dataset)[0]
    
    # check unique value binar classificaiton -> only 2 value
    valUnique = dataset[target].unique()
    
    # total data
    totalData = len(dataset.index)
    threshold = 0.3 * totalData
    
    if len(valUnique) <= 2:
        sumData1 = (dataset[target] == valUnique[0]).sum()
        sumData2 = (dataset[target] == valUnique[1]).sum()
        
        if (sumData1 <= threshold) or (sumData2 <= threshold):
            balanceDataset = SamplingData(dataset)
            balanceDataset.to_csv(app.config['UPLOAD_FOLDER'] + filename, index=False)
        else:
            balanceDataset = dataset

        return balanceDataset  


# Fungsi pembagian data menjadi data latih dan data uji
def SplitData(dataset):
    namaKelas = GetInfo(dataset)[0]

    atribut = dataset.columns.tolist()
    atribut = [data for data in atribut if data not in namaKelas]

    # X: data atribut dan Y: data kelas/target
    X = dataset[atribut]
    y = dataset[namaKelas]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    return result


# Fungsi untuk membangun jaringan ANN
def build_model_extra_args(jumlahInput, hp):
  
    # tuning learning rate
    hp_learning_rate= hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])

    # tuning neuron
    hp_units= hp.Int('units_0', min_value= 32, max_value= 704, step= 32)

    # input layer
    model= Sequential()
    model.add(Dense(units= hp_units, input_dim= jumlahInput, activation= 'relu'))

    # Tuning banyaknya hidden layer layer dan neuron tiap layer
    for i in range(hp.Int('num_layers', 2, 5)):
        
        # Tuning jumlah neuron tiap hidden layer
        hp_units = hp.Int('units_' + str(i), min_value= 32, max_value= 704, step= 32)
        model.add(Dense(units= hp_units, activation= 'relu'))

        # Tuning dropout tiap hidden layer
        hp_dropout = hp.Float('rate', min_value= 0.0, max_value= 0.8, step= 0.2)
        model.add(Dropout(hp_dropout))
    
    # Output layer
    model.add(Dense(1, activation= 'sigmoid'))
    
    # compile model
    model.compile(optimizer= Adam(learning_rate= hp_learning_rate), 
                  loss= 'binary_crossentropy', metrics= ['accuracy'])

    return model


# Get metric performance
def TunerSearchSummary(maxTrial):

    bestAccuracy = 0
    bestValAccuracy = 0

    for trial in range(maxTrial):
        filePath= app.config['RESULT_SUMMARY'] + '/trial_' + str(trial) + '/trial.json'
    
        # open JSON file
        fileJSON = open(filePath)
        
        # load data JSON
        data = json.load(fileJSON)
    
        accuracy = (data['metrics']['metrics']['accuracy']['observations'][0]['value'][0]) * 100
        val_accuracy = (data['metrics']['metrics']['val_accuracy']['observations'][0]['value'][0]) * 100

        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestValAccuracy = val_accuracy

    return [bestAccuracy, bestValAccuracy]


def Tuner(dataset):
    create_model = partial(build_model_extra_args, GetInfo(dataset)[1])

    # define max trials
    max_trials = 5

    tuner = BayesianOptimization (
        create_model,
        objective= 'accuracy',
        max_trials= max_trials,
        directory= 'result',
        project_name= 'bayesian',
        overwrite= True
    )

    return [tuner, max_trials]

def EarlyStopper():
    # inisialisasi Earlystopping untuk menghentikan iterasi ketika tidak terjadi peningkatan akurasi
    earlystopper = EarlyStopping(
        monitor = 'val_loss', 
        min_delta = 0, 
        patience = 10, 
        verbose= 1
    )

    return earlystopper


# Fungsi evaluasi model hasil training
def Evaluasi(model, dataset, history):

    X_test = SplitData(dataset)['X_test']
    y_test = SplitData(dataset)['y_test']

    # prediksi menggunakan data test
    y_predict = (model.predict(X_test) > 0.5).astype('int32')

    # confusion matrix
    confusionMatrix = confusion_matrix(y_test, y_predict)

    # data confusion matrix
    tp, fn, fp, tn = confusionMatrix.reshape(-1)
    accuracy = ((tn+tp)/(tn+tp+fp+fn)) * 100
    precision = (tp/(tp+fp)) * 100
    recall = (tp/(tp+fn)) * 100
    specificity = (tn/(tn+fp)) * 100
    error = 100 - accuracy

    # epoch
    history_dict = history.history
    epoch = len(history_dict['accuracy'])

    result = {
        'epoch': epoch,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'error': error
    }
    
    return result


if (__name__ == '__main__'):
    app.run(debug=True)

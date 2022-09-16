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
from unittest import result
import keras_tuner
import numpy as np
import os
import pandas as pd
import pickle
import scikeras
import shutil
from flask import Flask, jsonify, render_template, request, send_file
from functools import partial
from imblearn.combine import SMOTEENN
from joblib import dump
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# inisialisasi file template ada di folder template berisi file html
app = Flask(__name__, template_folder='template')

# inisialisasi folder untuk menyimpan dataset yang disubmit user
app.config['UPLOAD_FOLDER'] = './static/upload/'

# inisialisasi folder untuk menyimpan file-file history selama tuning
app.config['RESULT_SUMMARY'] = 'result/bayesian/'

# inisialisasi folder untuk menyimpan data hasil tuning hyperparameter (berupa data JSON)
app.config['RESULT_BEST_HPS'] = 'result/hyperparameter/'

# inisialisasi folder untuk menyimpan file hasil tuning (file tuning nya)
app.config['RESULT_FILE_TUNING'] = 'result/hyperparameter/'

# inisialisasi folder untuk menyimpan file model hasil training
# model yang bisa diunduh user ada di folder ini
app.config['RESULT_MODEL'] = 'result/model/'



# membuat folder di dalam server berdasarkan path yang dikirimkan
def CreateDirectory(path):
    dir = path
    if os.path.exists(dir):
        shutil.rmtree(dir)

    os.makedirs(dir)



# route untuk menampilkan halaman awal (index.html) -> HOME
@app.route('/')
def Home():

    # membuat folder result, result/hyperparameter, dan result/model pada server
    # struktur folder yaitu sebagai berikut:
    # result
    # ----- hyperparameter
    # ----- model

    CreateDirectory('result')
    CreateDirectory('result/hyperparameter')
    CreateDirectory('result/model')

    # menampilkan file index.html -> HOME
    return render_template('index.html')



# route untuk upload dataset dan membuka halaman tuning
@app.route('/upload', methods=['GET', 'POST'])
def Upload():
    # mengecek apakah ada method POST dari AJAX
    if request.method == 'POST':

        # mengecek apakah ada file yang dikirimkan melalui AJAX
        if request.files:
            fileDataset = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fileDataset.filename)
            fileDataset.save(file_path)

    # menampilkan file tuning.html
    return render_template('tuning.html')



# Fungsi untuk menampilkan semua kolom pada dataset. User diminta memilih kelas target
@app.route('/target', methods=['POST'])
def ChooseTargetClass():

    # mengecek apakah ada method POST dari AJAX
    if request.method == 'POST':

        # get informasi yang dikirimkan AJAX berupa nama dataset
        getData = request.get_data()
        filename = getData.decode('utf-8')

        # menentukan file path dari dataset
        path = app.config['UPLOAD_FOLDER'] + filename

        # membaca dataset menggunakan library pandas
        dataset = pd.read_csv(path)

        # list kolom
        listKolom = dataset.columns.tolist()

        result = {
            'list-kolom': listKolom,
            'file-path': path,
            'file-name': filename
        }

        return jsonify(result)



# route untuk mengecek dataset sebelum proses tuning hyperparameter
# cek dataset meliputi 3 kriteria yaitu:
# 1. IsAnyMissingValue() -> mengecek apakah ada data kosong? -> harus bernilai FALSE
# 2. IsAllNumeric() -> mengecek apakah semua data bertipe numeric? -> harus bernilai TRUE
# 3. IsBinaryClassification() -> mengecek apakah kelas target bernilai 2 value unik (binary classification) -> harus bernilai TRUE
# Output: hasil cek dataset
@app.route('/tuning/check', methods=['POST'])
def CheckDataset():

    # mengecek apakah ada method POST dari AJAX
    if request.method == 'POST':

        # get informasi yang dikirimkan AJAX berupa file path dan target class
        filePath = request.json.get('file_path')
        targetClass = request.json.get('target_class')

        # membaca dataset menggunakan library pandas
        dataset = pd.read_csv(filePath)

        # mengetahui jumlah data dalam dataset (sebelum sampling)
        jumlahDataSebelumSampling = len(dataset.index)

        # lakukan cek dataset meliputi 3 hal
        isAnyMissingValue = IsAnyMissingValue(dataset)
        isAllNumeric = IsAllNumeric(dataset)
        isBinaryClassification = IsBinaryClassification(dataset, targetClass)

        # buat pesan jika ada salah satu dari 3 hal yang tidak terpenuhi
        listMessage = []

        if isAnyMissingValue == True:
            message = 'Your dataset contains empty/missing values'
            listMessage.append(message)

        if isAllNumeric == False:
            message = 'Not all of your data is numeric type. Make sure that all your data is numeric type'
            listMessage.append(message)

        if isBinaryClassification == False:
            message = 'The target class you choose is not a case of binary classification'
            listMessage.append(message)


        # proses lakukan balancing data ketika 3 hal di atas terpenuhi
        # jika ada 1 hal yang tidak terpenuhi, proses tidak bisa berlanjut

        if (isAnyMissingValue == False) and (isAllNumeric == True) and (isBinaryClassification == True):

            # proses balancing data
            dataset = SamplingData(dataset, filePath, targetClass)

            # mengetahui jumlah data dalam dataset (setelah sampling)
            jumlahDataSetelahSampling = len(dataset.index)

            # status 1 -> dataset layak/ memenuhi syarat untuk proses tuning
            status = 1

            # inisialisasi dan assign hasil dari cek dataset untuk dikirimkan ke front-end melalui AJAX
            result = {
                'jumlah-data-sebelum-sampling': jumlahDataSebelumSampling,
                'jumlah-data-setelah-sampling': jumlahDataSetelahSampling,
                'jumlah-atribut':  GetJumlahAtribut(dataset),
                'status': status
            }

        else:
            # status 0 -> dataset tidak layak/ tidak memenuhi syarat untuk proses tuning
            status = 0 # tidak memenuhi syarat

            # kembalikan listMessage
            result = {
                'status': status,
                'message': listMessage
            }

            
        return jsonify(result)



# Fungsi utama untuk proses Tuning Hyperparameter
# Output: evaluasi hasil tuning
@app.route('/tuning/process', methods=['POST'])
def TuningHyperparameter():

    # mengecek apakah ada method POST dari AJAX
    if request.method == 'POST':

        # get informasi yang dikirimkan AJAX berupa nama dataset
        getData = request.get_data()
        filePath = getData.decode('utf-8')

        # membaca dataset menggunakan library pandas
        dataset = pd.read_csv(filePath)

        # split data
        X_train = SplitData(dataset)['X_train']
        y_train = SplitData(dataset)['y_train']

        # membuat konfigurasi jaringan
        create_model = partial(BuildModel, GetJumlahAtribut(dataset))

        # set maksimal percobaan sebanyak 5 kali
        max_trials = 5

        # konfigurasi tuner menggunakan bayesian
        tuner = BayesianOptimization (
            create_model,
            objective= 'accuracy',
            max_trials= max_trials,
            directory= 'result',
            project_name= 'bayesian',
            overwrite= True
        )

        # proses tuning
        tuner.search(X_train, y_train, epochs = 300, validation_split = 0.2, callbacks = [EarlyStopper()])

        # print hyperparameter paling optimal
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

        # simpan file tuner
        pickle.dump(tuner, open(app.config['RESULT_FILE_TUNING'] + 'tuner.pkl', "wb"))

        # save best hyperparameter
        pickle.dump(best_hps, open(app.config['RESULT_BEST_HPS'] + 'hyperparameter.pkl', "wb"))

        best_hps_values = best_hps.values
        best_hps_values['unit_input'] = GetJumlahAtribut(dataset)
        best_hps_values['accuracy'] = TuningResult(max_trials)[0]
        best_hps_values['val_accuracy'] = TuningResult(max_trials)[1]

        return jsonify(best_hps_values)


# Fungsi untuk membuat model (training model) menggunakan hyperparameter hasil tuning
# Input: dataset
# Output: hasil evaluasi
@app.route('/build', methods=['POST'])
def BuildModel():
    if request.method == 'POST':

        # get data dari AJAX
        getData = request.get_data()

        # mengetahui path/lokasi dataset
        filePath = getData.decode('utf-8')

        # baca dataset berdasarkan path menggunakan library pandas
        dataset = pd.read_csv(filePath)

        # load file tuner
        tuner = pickle.load(open(app.config['RESULT_FILE_TUNING'] + 'tuner.pkl', "rb"))

        # load best hyperparameter
        best_hps = pickle.load(open(app.config['RESULT_BEST_HPS'] + 'hyperparameter.pkl', "rb"))

        # split data
        X_train = SplitData(dataset)['X_train']
        y_train = SplitData(dataset)['y_train']

        # fit model menggunakan hasil tuning hyperparameter dan melatihnya
        model = tuner.hypermodel.build(best_hps)

        # mencatat riwayat hasil training selama proses training berlangsung
        history = model.fit(X_train, y_train, epochs= 500, validation_split= 0.2, callbacks= [EarlyStopper()])
        
        # save model
        model.save(app.config['RESULT_MODEL'] + "model.h5")

        # hasil evaluasi
        hasilEvaluasi = Evaluasi(model, dataset, history)

        return hasilEvaluasi


# Fungsi yang dijalankan ketika user klik tombol download model
# Output: file model.h5 yang bisa di download user
@app.route('/download')
def download():
    return send_file(app.config['RESULT_MODEL'] + 'model.h5', as_attachment = True)



# Fungsi untuk mengetahui jumlah kolom atribut
# Input: dataset
# Output: jumlah atribut (integer)
def GetJumlahAtribut(dataset):
    jumlahAtribut = len(dataset.columns) - 1
    return jumlahAtribut



# Fungsi untuk sampling data -> menangani data tidak seimbang
# Output: dataset hasil balancing
def SamplingData(dataset, filePath, targetClass):
    # menghitung total data
    totalData = len(dataset.index)

    # membuat batas threshold adalah 30% dari total data
    threshold = 0.3 * totalData

    # mengambil kelas target
    target = targetClass
    valUnique = dataset[target].unique()
    sumData1 = (dataset[target] == valUnique[0]).sum()
    sumData2 = (dataset[target] == valUnique[1]).sum()

    # jika banyak kelas minoritas < 30%, maka lakukan sampling
    if (sumData1 <= threshold) or (sumData2 <= threshold):

        # membagi data atribut dengan data target
        cols = dataset.columns.tolist()
        cols = [c for c in cols if c not in [target]]

        # X: data atribut dan Y: data kelas/target
        X = dataset[cols]
        Y = dataset[target]

        # jika tidak ada nilai kosong, maka proses sampling bisa dilakukan
        if not IsAnyMissingValue:
            sm =  SMOTEENN(random_state= 42)
            X_smote, Y_smote = sm.fit_resample(X, Y)

            # menggabungkan kelas atribut dengan kelas target
            X_smote[target] = Y_smote
            dataset = X_smote

            # replace dataset lama dengan dataset baru hasil sampling
            dataset.to_csv(filePath, index=False)

    return dataset



# Fungsi untuk mengecek apakah ada data kosong pada dataset
# Ouptut: True -> jika ada nilai kosong; False -> jika tidak ada nilai kosong
def IsAnyMissingValue(dataset):
    # proses sampling dengan SMOTE
    dataKosong = int(dataset.isna().sum().sum())

    if dataKosong == 0:
        return False
    else:
        return True


# Fungsi untuk mengecek apakah data pada dataset bertipe numeric semua atau tidak
# Input: dataset
# Output: True -> jika ada nilai semuanya numeric; False -> jika ada tipe data lain (e.g. string/date/boolean/etc)
def IsAllNumeric(dataset):
    dataTypeAllColumn = dataset.applymap(np.isreal).all().tolist()
    
    if False in dataTypeAllColumn:
        return False
    else:
        return True
    


# Fungsi mengecek apakah dataset merupakan kasus binary classification atau bukan
# Output: True -> Binary classificaiton; False -> bukan Binary Classification
def IsBinaryClassification(dataset, targetClass):
    # mengambil kelas target
    target = targetClass
    
    # cek data unik pada kelas target
    # jika hanya ada 2 value unik, maka binary classification
    valUnique = dataset[target].unique()

    # jika ya binary classification, kembalikan nilai True
    if len(valUnique) <= 2:
        return True
    else:
        return False



# Fungsi pembagian data menjadi data latih dan data uji
# Output: Data X (atribut) dan data y (target) traning dan testing
def SplitData(dataset, targetClass):
    target = targetClass

    atribut = dataset.columns.tolist()
    atribut = [data for data in atribut if data not in [target]]

    # X: data atribut dan Y: data kelas/target
    X = dataset[atribut]
    y = dataset[target]

    # dataset dibagi menjadi 80% data uji dan 20% data latih
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    return result



# Fungsi untuk membangun jaringan ANN
# Output: model hasil tuning
def BuildModel(jumlahInput, hp):
  
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



# Fungsi untuk menghentikan proses iterasi pada TuningHyperparameter() ketika tidak terjadi peningkatan akurasi
# Output: fungsi earlystopper
def EarlyStopper():
    earlystopper = EarlyStopping(
        monitor = 'val_loss', 
        min_delta = 0, 
        patience = 10, 
        verbose= 1
    )

    return earlystopper



# Fungsi untuk mengetahui akurais dan val akurasi saat proses tuning
# Output: n1ilai akurasi dan val akurasi
def TuningResult(maxTrial):

    # inisialisasi nilai akurasi terbaik dan nilai val akurasi terbaik
    bestAccuracy = 0
    bestValAccuracy = 0

    # proses loop sebanyak max trial yaitu 5 untuk mencari nilai akurasi dan val akurasi terbaik
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



# Fungsi evaluasi model hasil training
# Input: file model, dataset, dan history training
# Output: hasil evaluasi (banyaknya epoch, akurasi, presisi, recall, specificity, dan nilai error)
def Evaluasi(model, dataset, history):

    # membagi data menjadi data atribut test dan data target test
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
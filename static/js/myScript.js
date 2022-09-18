// General Configuration
$(document).ready(function() {
    Prism.highlightAll();
});

function makeTimer(startDate) {
    // var endTime = new Date("29 April 2018 9:56:00 GMT+01:00");	
    var endTime = startDate;
    endTime = (Date.parse(endTime) / 1000);

    var now = new Date();
    now = (Date.parse(now) / 1000);

    var timeLeft = now - endTime;

    var days = Math.floor(timeLeft / 86400);
    var hours = Math.floor((timeLeft - (days * 86400)) / 3600);
    var minutes = Math.floor((timeLeft - (days * 86400) - (hours * 3600)) / 60);
    var seconds = Math.floor((timeLeft - (days * 86400) - (hours * 3600) - (minutes * 60)));

    if (hours < "10") { hours = "0" + hours; }
    if (minutes < "10") { minutes = "0" + minutes; }
    if (seconds < "10") { seconds = "0" + seconds; }

    $(".execution-time").html('Execution time: ' + minutes + " minutes " + seconds + " seconds");
}


function TotalTimeExecution(dt2, dt1) {
    var diff = (dt2.getTime() - dt1.getTime()) / 1000;
    minutes = diff / 60;
    seconds = diff % 60;
    message = ": " + parseInt(minutes) + " minutes " + parseInt(seconds) + " seconds";

    return message;
}

function TimeExecution(status) {
    message = "";

    if (status == 1) {
        startDate = new Date();
        interval = setInterval(function() { makeTimer(startDate); }, 1000);
    } else {
        endDate = new Date();
        message = TotalTimeExecution(endDate, startDate)
        clearInterval(interval);
    }

    return message;
}

function DatasetAccepted() {
    $('#table-check-dataset-success').removeClass('hide');
    $('#btn-try-again').hide();
    $('#btn-tuning-dataset').show();

    statusDatasetHTML = "<span class='badge rounded-pill bg-info fw-bold'><i class='fa-sharp fa-solid fa-circle-check'></i> Yeay, your dataset is qualified to process</span>";
    $('#status-success').html(': ' + statusDatasetHTML);
}

function DatasetNotAccepted() {
    $('#table-check-dataset-error').removeClass('hide');
    $('#btn-try-again').show();
    $('#btn-tuning-dataset').hide();

    statusDatasetHTML = "<span class='badge rounded-pill bg-danger fw-bold'><i class='fa-solid fa-circle-xmark'></i> Sorry, your dataset isn't qualified to process</span>";
    $('#status-error').html(': ' + statusDatasetHTML);
}

// True -> loading (active)
// False -> finished (passive)

// function ketika memilih kelas target
function LoadingChooseTarget(status) {
    $('#btn-upload-dataset').addClass('hide');
    $('.choose-target').removeClass('hide');

    if (status == 1) {
        message = TimeExecution(1);
        $('#loader-choose-target').show();
        $('#div-choose-target').hide();
    } else {
        message = TimeExecution(0);
        $('#choose-target-execution-time').html(message);
        $('#loader-choose-target').hide();
        $('#div-choose-target').show();
    }
}


// function ketika loading dataset
function LoadingCheckDataset(status) {
    $('#btn-check-dataset').addClass('hide');
    $('.check-dataset').removeClass('hide');

    if (status == 1) {
        message = TimeExecution(1);
        $('#loader-check-dataset').show();
        $('#div-check-dataset').hide();
    } else {
        message = TimeExecution(0);
        $('#check-dataset-execution-time-success').html(message);
        $('#check-dataset-execution-time-error').html(message);
        $('#loader-check-dataset').hide();
        $('#div-check-dataset').show();
    }
}

// function ketika tuning
function LoadingTuning(status) {
    $('#btn-tuning-dataset').addClass('hide');
    $('.tuning-dataset').removeClass('hide');

    if (status == 1) {
        message = TimeExecution(1);
        $('#loader-tuning').show();
        $('#div-tuning').hide();
    } else {
        message = TimeExecution(0);
        $('#tuning-execution-time').html(message);
        $('#loader-tuning').hide();
        $('#div-tuning').show();
    }
}

// function ketika build model
function LoadingBuildModel(status) {
    $('#btn-build-model').addClass('hide');
    $('#btn-reset').addClass('hide');
    $('.build-model').removeClass('hide');

    if (status == 1) {
        message = TimeExecution(1);
        $('#loader-build-model').show();
        $('#div-build-model').hide();
    } else {
        message = TimeExecution(0);
        $('#build-execution-time').html(message);
        $('#loader-build-model').hide();
        $('#div-build-model').show();
        $('#card-manual').removeClass('hide');
    }
}


Dropzone.options.myAwesomeDropzone = {
    url: '/upload',
    autoProcessQueue: false,
    parallelUploads: 1,
    maxFilesize: 25, // MB
    uploadMultiple: false,
    acceptedFiles: ".csv",
    init: function() {
        var mydrop = this; // Closure

        // This is the event listener that triggers the start of the upload
        $('#btn-upload-dataset').on('click', function() {
            mydrop.processQueue();
        });

        this.on('success', function(file) {

            // Continue processing the queue if there are still files pending
            if (this.getQueuedFiles().length > 0) {
                this.processQueue();
            }

            $.ajax({
                url: '/target',
                type: 'POST',
                data: file.name,
                dataType: 'json',
                contentType: "application/json; charset=UTF-8",
                beforeSend: LoadingChooseTarget(1),
                success: function(response) {
                    fileName = response['file-name'];
                    filePath = response['file-path'];
                    listKolom = response['list-kolom'];

                    kodeHTMLHeaderTag = '<div class="select"><select name="choose-target" id="select-choose-target"><option selected disabled>Choose Target Class</option>';

                    $.each(listKolom, function(key, value) {
                        kodeHTMLHeaderTag += '<option value="' + value + '">' + value + '</option>'
                    });

                    kodeHTMLFooterTag = '</select>'

                    kodeHTML = kodeHTMLHeaderTag + kodeHTMLFooterTag;

                    $('#list-choose-target').html(kodeHTML);
                    $('#file-path').val(filePath);
                    $('#file-name').val(fileName);

                    console.log('AJAX Upload Berhasil Response');
                },
                error: function(error) {
                    console.log("AJAX Upload Gagal Response");
                    console.log(error);
                },
                complete: function() {
                    LoadingChooseTarget(0);
                }
            })

        });
    }
};


$('#btn-check-dataset').on('click', function(e) {
    e.preventDefault();

    fileName = $('#file-name').val();
    filePath = $('#file-path').val();
    targetClass = $('#select-choose-target').val();
    $('#target-class').val(targetClass);

    data = { file_path: filePath, target_class: targetClass }

    if (targetClass == null) {
        $('#warning-choose-target').removeClass('hide');
    } else {
        $('#warning-choose-target').addClass('hide');

        $.ajax({
            url: '/tuning/check',
            type: 'POST',
            data: JSON.stringify(data),
            dataType: 'json',
            contentType: "application/json; charset=UTF-8",
            beforeSend: LoadingCheckDataset(1),
            success: function(response) {
                statusDataset = response['status'];

                if (statusDataset == 1) {
                    jumlahDataSebelumSampling = response['jumlah-data-sebelum-sampling'];
                    jumlahDataSetelahSampling = response['jumlah-data-setelah-sampling'];
                    jumlahAtribut = response['jumlah-atribut'];

                    $('#nama-dataset-success').text(': ' + fileName);
                    $('#jumlah-data-sebelum-sampling-success').html(': ' + jumlahDataSebelumSampling + ' data');
                    $('#jumlah-data-setelah-sampling-success').html(': ' + jumlahDataSetelahSampling + ' data');
                    $('#jumlah-atribut-success').html(': ' + jumlahAtribut + ' Attribute');
                    $('#target-success').html(': ' + targetClass);

                    DatasetAccepted();
                } else {
                    listMessage = response['message'];

                    $('#nama-dataset-error').text(': ' + fileName);
                    $('#target-error').html(': ' + targetClass);

                    $.each(listMessage, function(key, value) {
                        errorDatasetHTML = "<tr><td colspan='2'><i class='fa-solid fa-triangle-exclamation text-warning'></i> " + value + "</td></tr>"
                        $('#table-check-dataset-error').append(errorDatasetHTML);
                    });

                    DatasetNotAccepted();
                }
            },
            error: function(error) {
                console.log("AJAX Cek Dataset Gagal Response");
                console.log(error);
            },
            complete: function() {
                LoadingCheckDataset(0);
            }
        })
    }
});


// AJAX
$("#btn-tuning-dataset").click(function(e) {
    e.preventDefault();

    filePath = $('#file-path').val();
    targetClass = $('#select-choose-target').val();
    $('#target-class').val(targetClass);

    data = { file_path: filePath, target_class: targetClass }

    // AJAX untuk kirim data dan menerima data kembalian dari app.py
    $.ajax({
        url: '/tuning/process',
        type: 'POST',
        data: JSON.stringify(data),
        dataType: 'json',
        contentType: "application/json; charset=UTF-8",
        beforeSend: LoadingTuning(1),
        success: function(response) {
            neuronInputLayer = String(response['unit_input']);
            neuronOutputLayer = "1";
            hiddenLayer = response['num_layers'];

            neuronHiddenLayer = "";
            listNeuronHiddenLayer = [];

            for (var i = 0; i < hiddenLayer; i++) {
                dataUnit = 'units_' + String(i);
                dataNeuron = String(response[dataUnit]);
                listNeuronHiddenLayer.push(dataNeuron);
                neuronHiddenLayer = neuronHiddenLayer + dataNeuron + " - ";
            }

            configuration = neuronInputLayer + " - " + neuronHiddenLayer + neuronOutputLayer;
            learningRate = String(response['learning_rate']);
            dropout = response['rate'].toFixed(1);
            accuracy = response['accuracy'].toFixed(2);
            valAccuracy = response['val_accuracy'].toFixed(2);

            $('#network-configuration').html(': ' + configuration);
            $('#input-layer').html(': 1 Layer');
            $('#hidden-layer').html(': ' + String(hiddenLayer) + ' Layers');
            $('#output-layer').html(': 1 Layer');
            $('#learning-rate').html(': ' + learningRate);
            $('#dropout').html(': ' + dropout + ' Rate');
            $('#accuracy').html(': ' + accuracy + ' %');
            $('#val-accuracy').html(': ' + valAccuracy + ' %');

            HyperparameterCopyToUser(neuronInputLayer, hiddenLayer, listNeuronHiddenLayer, learningRate, 32, dropout);
        },
        error: function(error) {
            console.log("AJAX Tuning Hyperparameter Gagal Response");
            console.log(error);
        },
        complete: function() {
            LoadingTuning(0);
        }
    })
});


$("#btn-build-model").click(function(e) {
    e.preventDefault();

    filePath = $('#file-path').val();
    targetClass = $('#select-choose-target').val();
    $('#target-class').val(targetClass);

    data = { file_path: filePath, target_class: targetClass }

    // AJAX untuk kirim data dan menerima data kembalian dari app.py
    $.ajax({
        url: '/build',
        type: 'POST',
        data: JSON.stringify(data),
        dataType: 'json',
        contentType: "application/json; charset=UTF-8",
        beforeSend: LoadingBuildModel(1),
        success: function(response) {
            epoch = response['epoch'];
            buildAccuracy = response['accuracy'].toFixed(2);
            buildRecall = response['recall'].toFixed(2);
            buildSpecificity = response['specificity'].toFixed(2);
            buildError = response['error'].toFixed(2);

            $('#epoch').html(': ' + epoch + ' / 500 epochs');
            $('#build-accuracy').html(': ' + buildAccuracy + ' %');
            $('#build-recall').html(': ' + buildRecall + ' %');
            $('#build-specificity').html(': ' + buildSpecificity + ' %');
            $('#build-error').html(': ' + buildError + ' %');
        },
        error: function(error) {
            console.log("AJAX Build Model Gagal Response");
            console.log(error);
        },
        complete: function() {
            LoadingBuildModel(0);
        }
    })
});

function HyperparameterCopyToUser(neuronInputLayer, numOfHiddenLayer, listNeuronHiddenLayer, learningRate, batchSize, dropout) {
    headerHTML = "<code class='language-py'># Import project library \nfrom keras.callbacks import EarlyStopping \nfrom keras.layers import Dense, Dropout, Flatten \nfrom keras.models import Sequential \nfrom tensorflow.keras.optimizers import Adam \n\n";
    bodyHTML = "def BuildModel() \n\t# structure of ANN network \n\tmodel = Sequential() \n";

    for (var i = 0; i < numOfHiddenLayer; i++) {
        if (i == 0) {
            bodyHTML += "\tmodel.add(Dense(" + listNeuronHiddenLayer[i] + ", input_dim = " + neuronInputLayer + ", activation = 'relu')) \n\tmodel.add(Dropout(" + dropout + "))\n";
        } else {
            bodyHTML += "\tmodel.add(Dense(" + listNeuronHiddenLayer[i] + ", activation = 'relu')) \n\tmodel.add(Dropout(" + dropout + "))\n";
        }
    }

    bodyHTML += "\tmodel.add(Dense(1, activation = 'sigmoid'))\n\n";

    footerHTML = "\toptimizer = Adam(learning_rate = " + learningRate.toString() + ") \n\tmodel.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy']) \n\n\t# Instance of class EarlyStopping() \n\t# Interupt running epoch when training performance not getting better \n\tearlystopper = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 1 ) \n\n\t# fit network to adjust weight on ANN network \n\tmodel.fit(X_train.values, y_train.values, epochs = 500, batch_size = " + batchSize.toString() + ", validation_split = 0.20, verbose = 0, callbacks = [earlystopper]) \n\n\treturn model </code>";

    $('#kode-python').html(headerHTML + bodyHTML + footerHTML);
    Prism.highlightAll();
}
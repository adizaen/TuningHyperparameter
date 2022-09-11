// General Configuration
$(document).ready(function() {

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
    $('#table-header').addClass('bg-success');
    $('#btn-try-again').hide();
    $('#btn-tuning-dataset').show();
}

function DatasetNotAccepted() {
    $('#table-header').addClass('bg-danger');
    $('#table-header').addClass('text-white');
    $('#btn-try-again').show();
    $('#btn-tuning-dataset').hide();
}

// True -> loading (active)
// False -> finished (passive)

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
        $('#check-dataset-execution-time').html(message);
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
    $('.build-model').removeClass('hide');

    if (status == 1) {
        $('#loader-build-model').show();
        $('#div-build-model').hide();
    } else {
        $('#loader-build-model').hide();
        $('#div-build-model').show();
    }
}


Dropzone.options.myAwesomeDropzone = {
    url: '/tuning',
    autoProcessQueue: false,
    parallelUploads: 1,
    maxFilesize: 25, // MB
    uploadMultiple: false,
    acceptedFiles: ".csv",
    init: function() {
        var mydrop = this; // Closure

        // This is the event listener that triggers the start of the upload
        $('#btn-check-dataset').on('click', function() {
            mydrop.processQueue();
        });

        this.on('success', function(file) {

            // Continue processing the queue if there are still files pending
            if (this.getQueuedFiles().length > 0) {
                this.processQueue();
            }
            $.ajax({
                url: '/tuning/check',
                type: 'POST',
                data: file.name,
                dataType: 'json',
                contentType: "application/json; charset=UTF-8",
                beforeSend: LoadingCheckDataset(1),
                success: function(response) {
                    namaDataset = response['nama-dataset'];
                    filePath = response['file-path'];
                    jumlahData = response['jumlah-data'];
                    jumlahAtribut = response['jumlah-atribut'];
                    target = response['target'];
                    dataKosong = response['data-kosong'];
                    statusDataset = response['status']

                    $('#file-path').val(filePath);
                    $('#nama-dataset').html(': ' + namaDataset);
                    $('#jumlah-data').html(': ' + jumlahData + ' Data');
                    $('#jumlah-atribut').html(': ' + jumlahAtribut + ' Attribute');
                    $('#target').html(': ' + target);
                    $('#data-kosong').html(': ' + dataKosong + ' Data');

                    if (statusDataset == 1) {
                        DatasetAccepted();
                        kodeHTML = "<span class='badge rounded-pill bg-info fw-bold'><i class='fa-sharp fa-solid fa-circle-check'></i> Yeay, your dataset is qualified to process</span>";
                    } else {
                        DatasetNotAccepted();
                        kodeHTML = "<span class='badge rounded-pill bg-danger fw-bold'><i class='fa-solid fa-circle-xmark'></i> Sorry, your dataset isn't qualified to process</span>";
                    }

                    $('#status').html(': ' + kodeHTML);

                },
                error: function(error) {
                    console.log("AJAX Gagal Response");
                    console.log(error);
                },
                complete: function() {
                    LoadingCheckDataset(0);
                }
            })

        });
    }
};


// AJAX
$("#btn-tuning-dataset").click(function(e) {
    e.preventDefault();

    file_path = $('#file-path').val();

    // AJAX untuk kirim data dan menerima data kembalian dari app.py
    $.ajax({
        url: '/tuning/process',
        type: 'POST',
        data: file_path,
        dataType: 'json',
        contentType: "application/json; charset=UTF-8",
        beforeSend: LoadingTuning(1),
        success: function(response) {
            neuronInputLayer = String(response['unit_input']);
            neuronOutputLayer = "1";
            hiddenLayer = String(response['num_layers']);

            neuronHiddenLayer = "";

            for (var i = 0; i < response['num_layers']; i++) {
                dataUnit = 'units_' + String(i);
                dataNeuron = String(response[dataUnit]);
                neuronHiddenLayer = neuronHiddenLayer + dataNeuron + " - ";
            }

            configuration = neuronInputLayer + " - " + neuronHiddenLayer + neuronOutputLayer;
            learningRate = String(response['learning_rate']);
            dropout = response['rate'].toFixed(1);
            accuracy = response['accuracy'].toFixed(2);
            valAccuracy = response['val_accuracy'].toFixed(2);

            $('#network-configuration').html(': ' + configuration);
            $('#input-layer').html(': 1 Layer');
            $('#hidden-layer').html(': ' + hiddenLayer + ' Layers');
            $('#output-layer').html(': 1 Layer');
            $('#learning-rate').html(': ' + learningRate);
            $('#dropout').html(': ' + dropout + ' Rate');
            $('#accuracy').html(': ' + accuracy + ' %');
            $('#val-accuracy').html(': ' + valAccuracy + ' %');
        },
        error: function(error) {
            console.log("AJAX Gagal Response");
            console.log(error);
        },
        complete: function() {
            LoadingTuning(0);
        }
    })
});
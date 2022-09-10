// General Configuration
$(document).ready(function() {
    // $('.success-check-dataset').hide();
    // $('.failed-check-dataset').hide();
    // $('.tuning-dataset').hide();
    // $('.build-model').hide();
});

// $('#btn-check-dataset').click(function() {
//     $('.success-check-dataset').removeClass('hide');
// });

// $('#btn-check-dataset').click(function() {
//     $('.success-check-dataset').removeClass('hide');
// });

$('#btn-tuning-dataset').click(function() {
    $('.tuning-dataset').removeClass('hide');
});

$('#btn-build-model').click(function() {
    $('.build-model').removeClass('hide');
    setInterval();
});

// Execution time
setInterval(function() {
    var date = new Date();
    $('.execution-time').html(
        "Execution time: " + date.getMinutes() + " minutes " + date.getSeconds() + " seconds"
    );
}, 500);

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
                success: function(response) {
                    $('.success-check-dataset').removeClass('hide');

                    jumlahData = response['jumlah_data'];
                    jumlahAtribut = response['jumlah_atribut'];
                    target = response['target'];
                    dataKosong = response['data_kosong'];
                    statusDataset = response['status']

                    $('#jumlah_data').html(': ' + jumlahData + ' Data');
                    $('#jumlah_atribut').html(': ' + jumlahAtribut + ' Attribute');
                    $('#target').html(': ' + target);
                    $('#data_kosong').html(': ' + dataKosong + ' Data');

                    if (statusDataset == 0) {
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
            })

        });
        this.on('uploadprogress', function(file, parameter, bytes) {

        });
    }
};

$('#my-awesome-dropzone').on('success', function() {
    var args = Array.prototype.slice.call(arguments);
    var msg = 'sukses dong';

    // Look at the output in you browser console, if there is something interesting
    console.log(args);
    console.log(msg);
});

// AJAX
// $("#btn-check-dataset").click(function() {
//     // e.preventDefault();
//     // AJAX untuk kirim data dan menerima data kembalian dari app.py
//     $.ajax({
//             url: '/tuning/check',
//             type: 'POST',
//             dataType: 'json',
//             contentType: "application/json; charset=UTF-8"
//         })
//         .done(function(data) {
//             console.log("berhasil");
//             console.log(data);
//         })
//         .fail(function(err) {
//             console.log("gagal");
//             console.log(err);
//         })
// });
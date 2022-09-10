$(function() {
    $('#picture').on('click', function() {
        $('#fileinput').trigger('click');
    });
});

function openCity(evt, idName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(idName).style.display = "block";
    evt.currentTarget.className += " active";
}

function openDatasetUnggah(evt, idName) {
    document.getElementById(idName).style.display = "block";
}

function openHyperparameter(evt, idName) {
    document.getElementById(idName).style.display = "block";
}

function openDatasetURL(evt, idName) {
    document.getElementById(idName).style.display = "block";
}

function openHyperparameterURL(evt, idName) {
    document.getElementById(idName).style.display = "block";
}

function clearDataUnggah(evt, className) {
    tabcontent = document.getElementsByClassName("unggah");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
}

function clearDataURL(evt, className) {
    tabcontent = document.getElementsByClassName("url");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    document.getElementById("uploadfromurl").value = "";
}


Dropzone.options.myDropzone = {
    // Configuration options go here
};
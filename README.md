# Smart Tune

## Logo
![](static/img/Logo.png)

## Tentang Smart Tune
Smart Tune merupakan sebuah aplikasi berbasis web (web based application) yang digunakan untuk melakukan tuning hyperparameter pada algoritme Artificial Neural Network (ANN) dalam kasus binary classification. Smart Tune akan memberikan konfigurasi jaringan yang optimal menggunakan algortime Bayesian Optimization sehingga user dapat mengadopsi kode Python yang Smart Tune hasilkan untuk diterapkan pada project user. Selain itu, Smart Tune juga dilengkapi dengan fitur build model yang memungkinkan membuild model setelah proses tuning dijalankan. 

## Cara Menggunakan
* Upload Dataset
Pertama user diminta untuk upload dataset dalam bentuk file .csv. Ukuran maksimal yang diperbolehkan yaitu 25 Mb. User bisa drag and drop file dataset ke dalam dropzone yang disediakan, atau upload manual dengan cara klik dropzone. Dalam sekali upload, hanya diperbolehkan 1 file saja. Untuk memulai proses upload, user diminta klik tombol upload yang disediakan.
* Memilih Target
Proses kedua setelah user upload dataset yaitu memilih kelas target. User memilih kelas target dalam list yang telah disediakan oleh sistem. List akan berisi semua nama kolom dari dataset yang diupload oleh user. Kelas target adalah kolom/ atribut yang akan dilakukan klasifikasi. Sebagai contoh pada kasus dataset penyakit jantung. Kelas target yang bisa digunakan yaitu kolom yang menandakan pasien yang mengalami penyakit jantung atau tidak.
* Cek Dataset
Pada proses ini akan dilakukan 3 proses pengecekan dan 1 proses manipulasi dataset guna memastikan bahwa dataset yang diupload user qualified untuk dilakukan proses tuning. Proses tersebut yaitu: pengecekan nilai kosong, pengecekan tipe data, pengecekan kelas target, dan proses sampling.
* Tuning Hyperparameter
Proses ini merupakan proses utama dalam aplikasi Smart Tune. Di mana pada proses inilah terjadi tuning hyperparameter. Proses di mana Smart Tune mencari konfigurasi jaringan yang optimal untuk dataset yang diupload oleh user. Pada proses ini ada beberapa hyperparameter yang akan dicari konfigurasinya diantaranya yaitu menentukan konfigurasi jaringan, neuron tiap layer, dropout, dan learning rate.
* Build Model
Proses ini merupakan proses terakhir yang disediakan oleh Smart Tune. Build model merupakan proses melatih data berdasarkan nilai hyperparameter yang telah dilakukan sebelumnya. Proses ini akan menghasilkan file model.h5 yang dapat langsung digunakan user untuk memprediksi data baru dan bisa ditanamkan pada berbagai aplikasi sesuai kebutuhan.

## Batasan Aplikasi
* Diperuntukan untuk kasus binary classification.
* Dataset berupa data tabular bertipe data .csv.
* Hanya diperuntukan untuk project dengan algoritme ANN.
* Batasan file dataset yaitu 25 Mb. Semakin besar file, proses tuning semakin lama.
*Smart Tune berasumsi bahwa dataset yang diupload telah melewati proses cleaning (pre-processing data).

## Tools Yang Digunakan
Ada beberapa tools dan bahasa pemrograman dalam membangun aplikasi Smart Tune. Bahasa yang digunakan yaitu Python untuk backend. Sementara untuk front end menggunakan HTML, CSS, dan AJAX jQuery untuk pertukaran data. Smart Tune dibangun menggunakan framework Flask. Pengembang menggunakan text editor Visual Studio Code untuk membantu dalam proses pengembangan software.

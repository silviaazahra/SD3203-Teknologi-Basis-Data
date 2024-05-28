## **Tugas Teknologi Basis Data**

Nama : Silvia Azahrani

NIM : 121450070

Kelas : RB


### **Tiga Cara Dalam Menyimpan dan Mengakses Banyak Gambar Pada Python**

#### **Setup**
##### Setup Dataset

```python
import numpy as np
import pickle
from pathlib import Path
# Path to the unzipped CIFAR data
data_dir = Path("D:/Downloads/cifar-10-python/cifar-10-batches-py")
# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])
print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```
::: {.output .stream .stdout}
    Loaded CIFAR-10 training set:
     - np.shape(images)     (50000, 32, 32, 3)
     - np.shape(labels)     (50000,)

Kode diatas bertujuan untuk memuat dataset CIFAR-10, sebuah kumpulan data gambar yang digunakan untuk tujuan pengenalan objek dalam bidang vision komputer. Dataset ini terdiri dari 50.000 gambar dengan label yang sesuai, diambil dari 10 kategori objek yang berbeda. Melalui proses iteratif, kode membaca setiap batch file dataset, mengonversi data gambar yang awalnya ter-flatten menjadi bentuk yang dapat diinterpretasikan (32x32 piksel dengan 3 saluran warna), dan menyimpannya dalam array numpy. Label-label yang sesuai juga disimpan dalam array terpisah. Setelah proses ini selesai, informasi tentang bentuk array gambar dan label dicetak untuk memverifikasi bahwa data telah dimuat dengan benar. Dengan demikian, kode ini mempersiapkan dataset CIFAR-10 untuk penggunaan dalam pelatihan atau pengujian model.

##### Setup untuk menyimpan gambar dalam disk


Pada proses menyimpan gambar didalam disk, pelu dilakukan persiapan terhadap environment agar saving dan access gambar pada disk dapat dilakukan. Berikut merupakan kode untuk setup pada shell perangkat yang digunakan.

```python
$ pip install Pillow
$ conda install -c conda-forge pillow
```


##### Mempersiapkan LMDB

LMDB dipetakan dalam memori yang artinya, ia mengembalikan penunjuk langsung ke alamat memori dari kunci dan nilai, tanpa perlu menyalin apa pun di memori seperti yang dilakukan kebanyakan database lainnya.
Pada proses menyimpan gambar didalam LMDB, pelu dilakukan persiapan terhadap environment agar saving dan access gambar pada disk dapat dilakukan. Berikut merupakan kode untuk setup pada python shell perangkat yang digunakan.

```python
$ pip install lmdb
$ conda install -c conda-forge python-lmdb
```


##### Mempersiapkan HDF5

HDF5 adalah singkatan dari Hierarchical Data Format. File HDF terdiri dari dua jenis objek:
*   Dataset
*   Grup

Dataset adalah array multidimensi, dan grup terdiri dari kumpulan data atau grup lain. Array multidimensi dengan ukuran dan tipe apa pun dapat disimpan sebagai kumpulan data, namun dimensi dan tipenya harus seragam dalam kumpulan data. Setiap dataset harus berisi array berdimensi N yang homogen. Berikut merupakan kode untuk setup pada python shell perangkat yang digunakan untuk mempersiapkan HDF5.

```pyhon
$ pip install h5py
$ conda install -c conda-forge h5py
```



#### **Storing a Single Image**

Langkah awal yakni mempersiapkan folder berdasarkan setiap metode yang akan berisi file gambar database. Contoh yang digunakan adalah membandingkan kinerja antara jumlah file, dengan 1 gambar hingga 100.000 gambar. Karena lima kumpulan CIFAR-10 berjumlah 50.000 gambar, maka setiap gambar diproses dua kali untuk mendapatkan 100.000 gambar.

```python
from pathlib import Path
disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)
```
Kode diatas menggunakan modul `pathlib` untuk membuat direktori baru dalam struktur penyimpanan data. Tiga direktori baru dibuat: `disk_dir`, `lmdb_dir`, dan `hdf5_dir`, masing-masing menunjukkan tiga jenis penyimpanan data yang berbeda. Fungsi `mkdir()` digunakan untuk membuat direktori baru. Parameter `parents=True` memungkinkan pembuatan direktori bertingkat jika diperlukan, sedangkan `exist_ok=True` memastikan bahwa tidak ada kesalahan yang dihasilkan jika direktori tersebut sudah ada. Dengan demikian, kode ini mempersiapkan infrastruktur penyimpanan data yang diperlukan untuk berbagai jenis format data.


##### Menyimpan ke Disk


Kode tersebut berfungsi untuk menyimpan gambar dalam format .png ke dalam disk dan label gambar ke dalam file .csv. Fungsi ini menerima tiga parameter: image (array gambar dengan ukuran (32, 32, 3)), image_id (ID unik dalam bentuk integer untuk gambar tersebut), dan label (label gambar). Pertama, gambar dikonversi menjadi objek gambar menggunakan modul PIL (Python Imaging Library) dan kemudian disimpan ke dalam disk dengan menggunakan fungsi save. Selanjutnya, label gambar disimpan ke dalam file CSV dengan label ditulis pada baris pertama dari file tersebut.

```python
from PIL import Image
import csv
def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")
    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
```

Fungsi `store_single_disk` digunakan untuk menyimpan satu gambar dan labelnya ke dalam disk. Gambar disimpan dalam format .png dan label disimpan dalam file .csv. Dengan menggunakan ID unik untuk gambar, informasi gambar dan label terhubung secara terpisah namun terorganisir.

##### Menyimpan ke LMBD

Kode pertama bertujuan untuk memproses gambar dari dataset CIFAR dengan tujuan menciptakan kunci unik untuk masing-masing gambar. Saat objek dari kelas dibuat, metode __init__ digunakan dengan parameter gambar (dalam bentuk array) dan labelnya dari dataset CIFAR. Pada metode __init__, dimensi gambar disimpan untuk rekonstruksi kanal dan ukuran gambar, sementara gambar dikonversi menjadi format byte menggunakan metode tobytes() untuk kemudahan penyimpanan. Selain itu, label gambar juga disimpan. Untuk mengembalikan gambar dari representasi byte, langkah-langkah yang dijalankan adalah: bytes gambar dikonversi kembali menjadi array numpy menggunakan np.frombuffer dengan tipe data uint8. Array tersebut kemudian diubah dimensinya kembali ke bentuk aslinya menggunakan reshape, dengan mempertimbangkan dimensi dan kanal gambar yang telah disimpan sebelumnya.

```python
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.label = label
    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
```



Kode kedua  berfungsi untuk menyimpan gambar ke dalam basis data LMDB (Lightning Memory-Mapped Database). Fungsi menerima tiga parameter: image (array gambar dengan ukuran 32x32x3), image_id (ID unik gambar dalam bentuk integer), dan label (label gambar). Pertama, kode menghitung ukuran peta memori basis data LMDB berdasarkan ukuran gambar. Kemudian, sebuah lingkungan LMDB baru dibuat dengan menggunakan lmdb.open(), menentukan lokasi dan ukuran peta memori. Selanjutnya, transaksi tulis dimulai dengan env.begin(), di mana objek CIFAR_Image yang berisi gambar dan label diubah menjadi bentuk serial menggunakan modul pickle, dan disimpan dalam basis data LMDB menggunakan txn.put(). Kunci untuk entri basis data dihasilkan dari image_id dengan format tertentu. Setelah transaksi selesai, lingkungan LMDB ditutup dengan env.close().

```python
import lmdb
import pickle
def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10
    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)
    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
```



##### Menyimpan ke HDF5


Kode berikut bertujuan untuk menyimpan sebuah gambar ke dalam sebuah file HDF5. Fungsi ini menerima tiga parameter yakni image yang merupakan array gambar dengan ukuran (32, 32, 3), image_id yang merupakan ID unik dalam bentuk integer untuk gambar tersebut, dan label yang merupakan label dari gambar tersebut.

Pertama, sebuah file HDF5 baru dibuat menggunakan h5py.File() dengan menentukan lokasi dan nama file serta mode "w" untuk menulis. Selanjutnya, dataset gambar dan metadata (label) dibuat di dalam file tersebut menggunakan file.create_dataset(). Untuk dataset gambar, ukuran dan tipe datanya ditentukan berdasarkan ukuran dan tipe data dari gambar yang diberikan. Sedangkan untuk metadata (label), ukuran dan tipe data ditentukan berdasarkan ukuran dan tipe data dari label yang diberikan.

Setelah semua dataset dibuat, file HDF5 ditutup dengan menggunakan file.close().

```python
import h5py
def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")
    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
```



##### Eksperimen Storing Single Images


Berikut merupakan implementasi ketiga fungsi sebelumnya yang telah didefiniskan disimpan kedalam fugsi _stor_single_funcs untuk dapat dipanggil kembali saat melakukan penyimpanan data pertama CIFAR dan labelnya melalui tiga cara berbeda.

```python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```


```python
from timeit import timeit
store_single_timings = dict()
for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

:::{Output}
Method: disk, Time usage: 0.5281889000325464
Method: lmdb, Time usage: 0.6817705999710597
Method: hdf5, Time usage: 0.08505170000717044

#### **Storing Many Images**

##### Menyesuaikan Kode Untuk Many Images

Tiga fungsi baru telah dibuat untuk menyimpan beberapa gambar dalam berkas bertipe .png, dengan masing-masing tujuan yang berbeda. Pertama, fungsi store_many_disk bertujuan untuk menyimpan array gambar ke dalam disk. Fungsi ini menerima parameter images (array gambar dengan ukuran N, 32, 32, 3) dan labels (array label dengan ukuran N, 1), di mana N adalah jumlah gambar. Dalam fungsi ini, jumlah gambar yang akan disimpan dihitung, kemudian gambar-gambar disimpan satu per satu ke dalam disk dengan format PNG menggunakan loop for. Setiap gambar disimpan dengan nama file yang sesuai dengan indeks gambar, dan label-label gambar disimpan dalam file CSV yang nama file-nya sesuai dengan jumlah gambar, dengan setiap label ditulis ke dalam baris terpisah dalam file CSV.

Kedua, fungsi store_many_lmdb memiliki tujuan yang mirip dengan store_single_lmdb, yaitu menyimpan seluruh array gambar ke dalam sebuah basis data LMDB dalam satu transaksi tulis. 

Terakhir, fungsi store_many_hdf5 digunakan untuk menyimpan array gambar ke dalam sebuah berkas HDF5. Dalam fungsi ini, dataset "images" dibuat untuk menyimpan array gambar, dan dataset "meta" untuk menyimpan array label. Setelah dataset dibuat, berkas HDF5 ditutup.

```python
store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")
    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])
def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    map_size = num_images * images[0].nbytes * 10
    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)
    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")
    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
```



##### Mempersiapkan Dataset

Berikut proses mempersiapkan dataset sebanyak 1000 gambar.

```python
cutoffs = [10, 100, 1000, 10000, 100000]
# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)
# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```
::: {.output .stream .stdout}
    (100000, 32, 32, 3)
    (100000,)


##### Eksperimen Storing Many Images


Dalam proses implementasinya, sebuah dictionary _store_many_funcs diinisialisasi yang memuat fungsi-fungsi untuk menyimpan array gambar dan label menggunakan tiga metode berbeda: disk, lmdb, dan hdf5. Selanjutnya, sebuah dictionary store_many_timings dibuat untuk merekam waktu yang dibutuhkan untuk setiap metode penyimpanan.

Langkah berikutnya adalah iterasi melalui setiap nilai cutoff dalam variabel cutoffs. Pada setiap iterasi, dilakukan juga iterasi melalui tiga metode penyimpanan. Untuk setiap kombinasi cutoff dan metode, waktu eksekusi diukur menggunakan fungsi timeit untuk menentukan waktu yang diperlukan dari pemanggilan fungsi penyimpanan dengan parameter yang sesuai. Hasil waktu eksekusi kemudian dicatat ke dalam dictionary store_many_timings.

```python
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)
from timeit import timeit
store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}
for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)
        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.043001499999945736
    Method: lmdb, Time usage: 0.013635200000180703
    Method: hdf5, Time usage: 0.054849900000135676
    Method: disk, Time usage: 0.16676769999980934
    Method: lmdb, Time usage: 0.005678900000020803
    Method: hdf5, Time usage: 0.0023858000001837354
    Method: disk, Time usage: 1.6323519000000033
    Method: lmdb, Time usage: 0.03800269999987904
    Method: hdf5, Time usage: 0.005430300000170973
    Method: disk, Time usage: 12.146795200000042
    Method: lmdb, Time usage: 0.29728830000021844
    Method: hdf5, Time usage: 0.025982599999679223
    Method: disk, Time usage: 142.75358489999962
    Method: lmdb, Time usage: 4.296912700000121
    Method: hdf5, Time usage: 0.46554439999999886


#### **Reading a Single Image**

##### Membaca dari Disk

Berikut implementasi untuk membaca sebuah gambar tunggal beserta metadata dari file format .png dan .csv. Fungsi read_single_disk digunakan untuk melakukan tugas ini dengan menerima satu parameter, yaitu image_id yang merupakan ID unik dari gambar yang ingin dibaca.

Pertama, gambar dibuka dari file menggunakan Image.open() dari modul PIL. Gambar tersebut kemudian diubah menjadi array numpy menggunakan np.array(). Selanjutnya, metadata label dari gambar dibaca dari file CSV yang sesuai dengan ID gambar. Label ini diambil dari baris pertama file CSV.

Fungsi mengembalikan gambar dan label dalam bentuk tuple, di mana gambar direpresentasikan sebagai array numpy dengan ukuran 32, 32, 3, dan label direpresentasikan sebagai integer.

```python
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))
    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])
    return image, label
```



##### Membaca dari LMDB

Proses membaca gambar dan metadata dari basis data LMDB dimulai dengan membuka lingkungan LMDB dan memulai transaksi baca. Fungsi read_single_lmdb bertujuan untuk tugas ini dengan menerima satu parameter, yaitu image_id yang merupakan ID unik gambar yang ingin dibaca. Lingkungan LMDB dibuka dalam mode hanya baca (readonly=True) menggunakan lmdb.open(), dan transaksi baca dimulai dengan env.begin(). Di dalam transaksi tersebut, kunci yang sesuai dengan ID gambar diambil dalam bentuk string menggunakan txn.get(). Data yang diperoleh di-deserialize dengan pickle.loads(), karena data gambar disimpan dalam bentuk serial menggunakan modul pickle. Setelah objek CIFAR_Image berhasil diambil, gambar dan label diekstraksi, dan lingkungan LMDB ditutup sebelum fungsi mengembalikan gambar dan label dalam bentuk tuple, dengan gambar direpresentasikan sebagai array numpy 32x32x3 dan label sebagai integer.

```python
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)
    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()
    return image, label
```



##### Membaca dari HDF5

Proses membaca gambar dari file HDF5 dimulai dengan membuka file tersebut menggunakan h5py.File() dalam mode baca dan tulis ("r+"). Fungsi read_single_hdf5 bertujuan untuk tugas ini dengan menerima satu parameter, yaitu image_id yang merupakan ID unik gambar yang ingin dibaca. Gambar dan label kemudian dibaca dari dataset yang sesuai di dalam file HDF5, dengan dataset gambar diambil dari path "/image" dan dataset label dari path "/meta". Kedua dataset tersebut diubah menjadi array numpy menggunakan np.array() dengan tipe data "uint8" untuk memastikan konsistensi tipe data. Seperti sebelumnya, fungsi mengembalikan gambar dan label dalam bentuk tuple, dengan gambar direpresentasikan sebagai array numpy 32x32x3 dan label sebagai integer.

```python
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")
    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))
    return image, label
```



Selanjutnya membuat Dictionary _read_single_funcs dibuat untuk menampung tiga kunci yang masing-masing merepresentasikan sebuah metode penyimpanan data, yaitu disk, lmdb, dan hdf5. Setiap kunci mengacu pada sebuah fungsi yang telah didefinisikan sebelumnya untuk membaca data dari metode penyimpanan yang sesuai.

```python
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```



##### Eksperimen Reading a Single Image

Dalam proses pengukuran waktu untuk membaca sebuah gambar dan label, tiga metode penyimpanan yang berbeda digunakan: disk, lmdb, dan hdf5. Setiap metode ini diukur waktu eksekusinya menggunakan fungsi timeit pada setiap iterasi. Hasil waktu eksekusi dari masing-masing metode disimpan ke dalam dictionary read_single_timings.

```python
from timeit import timeit
read_single_timings = dict()
for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```



#### **Reading Many Images**

##### Menyesuaikan Kode Untuk Many Images

Dalam implementasi penerapan pembacaan beberapa file gambar sekaligus, digunakan tiga fungsi: read_many_disk, read_many_lmdb, dan read_many_hdf5. Setiap fungsi ini menerima satu parameter, yaitu num_images, yang menentukan jumlah gambar yang ingin dibaca. Fungsi read_many_disk membaca setiap gambar dari disk satu per satu, kemudian mengambil label-labelnya dari file CSV yang sesuai. read_many_lmdb membaca semua gambar dalam satu transaksi dari basis data LMDB dan mengambil label-labelnya. Sedangkan read_many_hdf5 membaca semua gambar dan label-labelnya dari file HDF5.

Hasil dari pembacaan gambar dan label dikembalikan dalam bentuk tuple images, labels, di mana images adalah array gambar dengan ukuran N, 32, 32, 3 dan labels adalah array label dengan ukuran N, 1, di mana N adalah jumlah gambar yang ingin dibaca. Dictionary _read_many_funcs digunakan untuk menyimpan fungsi-fungsi pembacaan dengan kunci yang sesuai dengan metode penyimpanan yang digunakan.

```python
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))
    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels
def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)
    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels
def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")
    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")
    return images, labels
_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)
```

##### Eksperimen Reading Many Images

Dalam kode implementasi ini, dilakukan pengukuran waktu untuk membaca sejumlah gambar dari tiga metode penyimpanan yang berbeda: disk, lmdb, dan hdf5. Setiap iterasi menggunakan fungsi timeit untuk mengukur waktu eksekusi saat memanggil fungsi pembacaan data sesuai metode penyimpanan yang digunakan. Parameter num_images yang diberikan kepada fungsi pembacaan menentukan jumlah gambar yang ingin dibaca, nilai ini diberikan oleh variabel cutoff. Hasil waktu eksekusi dari masing-masing metode penyimpanan untuk setiap jumlah gambar yang dibaca disimpan dalam dictionary read_many_timings.

```python
from timeit import timeit
read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}
for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)
        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, No. images: 10, Time usage: 1.9000144675374031e-06
    Method: lmdb, No. images: 10, Time usage: 2.500019036233425e-06
    Method: hdf5, No. images: 10, Time usage: 2.100015990436077e-06
    Method: disk, No. images: 100, Time usage: 2.200016751885414e-06
    Method: lmdb, No. images: 100, Time usage: 2.600019797682762e-06
    Method: hdf5, No. images: 100, Time usage: 2.200016751885414e-06
    Method: disk, No. images: 1000, Time usage: 1.600012183189392e-06
    Method: lmdb, No. images: 1000, Time usage: 1.700012944638729e-06
    Method: hdf5, No. images: 1000, Time usage: 2.0999577827751637e-06
    Method: disk, No. images: 10000, Time usage: 1.600012183189392e-06
    Method: lmdb, No. images: 10000, Time usage: 2.00001522898674e-06
    Method: hdf5, No. images: 10000, Time usage: 1.800013706088066e-06
    Method: disk, No. images: 100000, Time usage: 1.500011421740055e-06
    Method: lmdb, No. images: 100000, Time usage: 2.100015990436077e-06
    Method: hdf5, No. images: 100000, Time usage: 2.100015990436077e-06


#### **Considering Disk Usage**

HDF5 dan LMDB memerlukan lebih banyak ruang disk daripada menyimpan gambar biasa dalam format .png. Kinerja dan penggunaan disk dari kedua metode tersebut sangat dipengaruhi oleh berbagai faktor, termasuk sistem operasi dan ukuran data yang disimpan. LMDB mengoptimalkan efisiensinya melalui caching dan pemanfaatan ukuran halaman sistem operasi, sehingga dalam beberapa kasus, HDF5 sedikit lebih efisien dalam penggunaan disk daripada LMDB.

#### **Kesimpulan**

Artikel tersebut membahas tiga metode penyimpanan dan pengaksesan gambar menggunakan Python: penyimpanan dan pembacaan gambar dalam format .png di disk, penggunaan LMDB, dan penggunaan HDF5. Meskipun penyimpanan gambar sebagai file .png adalah pendekatan yang intuitif, metode seperti HDF5 atau LMDB menawarkan manfaat kinerja yang signifikan. Namun, tidak ada metode penyimpanan yang sempurna, dan pilihan harus disesuaikan dengan karakteristik dataset dan kebutuhan kasus penggunaan tertentu. Evaluasi kinerja menunjukkan perbedaan dalam penggunaan ruang disk dan efisiensi, dengan LMDB dan HDF5 menawarkan keuntungan tertentu tergantung pada skenario penggunaan dan ukuran data. Dalam keseluruhan penilaian, penting untuk mempertimbangkan keseimbangan antara ruang disk yang tersedia, kinerja, dan kebutuhan spesifik aplikasi ketika memilih metode penyimpanan yang sesuai.

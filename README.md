# ML-A11.2022.14595-UAS
Judul : Sentimne Analisis Review Film menggunakan Algoritma Naive Bayes<br>
Nama : Wahyu Aji <br>
Nim : A11.2022.14595<br>

## Proyek ini mengimplementasikan model pembelajaran mesin untuk sentimen Analis Teks Proccesing untuk mengungkap nilai positif dan negatif dalam sebuat reviw film.

## Instalasi
1. **clone Resposposi ** 
   ``` sh
   https://github.com/Wahyuajipamungkas/ML-A11.2022.14595-UAS
   ```
2. ** Instal pustaka yang diperlukan ** 
   ```sh
   pip install pandas nimpy matplotlib seaborn
   ```

## Deskripsi Proyek
Proyek ini terdiri dari berbagai langkah untuk menentukan hasil analisisnya yaitu pemisahan score negatif dan positif, pemisahan kalimat dan kata dan menentukan kata yang positif dan negatif

## Tujuan
proses yang bertujuan untuk memisahkan isi dalam database yang berbentuk teks bersifat positif dan negatif. Ulasan atau pendapat khalayak umum sangat penting bagi pengembangan film selanjutnya dan bagi film itu sendiri. Algoritma yang disarankan untuk melakukan percobaan ini adalah algoritma Naïve Bayen(BA) adalah Algoritma Machine Learning untuk masalah Sentimen. 

## Tahapan
<img src="images/Screenshot 2024-07-25 202852.png" align="center" width="800" height="250">

## Latar Belakang Masalah
Dalam era digital saat ini, internet telah menjadi tempat utama bagi 
individu untuk berbagi pendapat dan pengalaman mereka terkait berbagai hal, 
termasuk film. Dengan adanya platform seperti situs web ulasan film dan media 
sosial, masyarakat memiliki akses tak terbatas untuk mengekspresikan opini 
mereka tentang film yang mereka tonton.
 Namun, memahami sentimen kolektif masyarakat terhadap suatu film bisa 
menjadi tugas yang menantang dan memakan waktu jika dilakukan secara 
manual. Oleh karena itu, penggunaan algoritma machine learning, khususnya 
algoritma Naive Bayes, dapat menjadi solusi efisien untuk menganalisis dan 
mengekstraksi sentimen dari ulasan film yang tersebar luas di internet.
 Algoritma Naive Bayes adalah salah satu metode klasifikasi yang umum 
digunakan dalam analisis sentimen. Metode ini bekerja dengan mengasumsikan 
bahwa setiap fitur (kata atau frasa) dalam sebuah dokumen independen satu 
sama lain, meskipun demikian, metode ini telah terbukti cukup efektif dalam 
banyak aplikasi analisis sentimen, termasuk analisis ulasan film.

## Permasalahan
Meskipun ulasan film tersedia dalam jumlah besar di berbagai platform 
online, memahami sentimen umum yang terkandung di dalamnya merupakan 
tantangan yang kompleks dan memakan waktu. Tanpa alat yang sesuai, 
mengidentifikasi apakah suatu ulasan bersifat positif, negatif, atau netral 
memerlukan waktu dan tenaga manusia yang signifikan.
 Oleh karena itu, diperlukan sebuah solusi otomatis yang dapat menganalisis 
ulasan film secara cepat dan efisien. Dalam hal ini, masalah yang muncul adalah 
bagaimana mengimplementasikan algoritma analisis sentimen, khususnya 
algoritma Naive Bayes, untuk mengekstraksi dan mengklasifikasikan sentimen 
dari ulasan film yang tersebar luas di internet.

## Dataset
Link dataset : https://github.com/Wahyuajipamungkas/ML-A11.2022.14595-UAS/TugasAkhir<br>
Didalam dataset disney.csv itu terdapat 8000+ data yang dikumpulkan dalam beberapa tahun. didalam dataset tersebut terdapat atribut antara lain nama_film,film_year, author_name,review_date, Score, titlee_name, review_text, POU. disini saya menggunakan kolom score untuk dijadikan label. Saya mengambil dataset ini dari kaggle langsung dan data ini bersifat public.<br>

Untuk tahapannya/ ERDnya<br>
<img src="images/Screenshot 2024-07-25 202852.png" align="center" width="800" height="800">

## Proses Learning
1. Import Pustaka
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style("whitegrid")

#set Warning
import warnings
warnings.filterwarnings('ignore')

pd.pandas.set_option('display.max_columns',None)
```
2. Memasukkan Dataset
```Python
filename = "disney.csv"
db = pd.read_csv(filename, encoding='latin-1')
db.head()
```
Dalam dataset ini terdapat 8000+ data dan memiliki 8 Atribut<br>
3. Memilih Atribut 
```python
db.drop(columns=['film_name','film_year','title_name','author_name','review_date','POU'], inplace=True)
db.columns = ['score','review_text']
db.head()
```
Dalam memilih atribut ini saya menggunakan atribut score dan review_text untuk membuat sentimen analisis ini.<br>
4. Preprocessing Data<br>
a. Cleaning text
```python
import string
import re
def clean_text(text):
    return re.sub('[^a-zA-Z]', ' ', text).lower()
db['review_text'] = db['review_text'].fillna('')
db['cleaned_text'] = db['review_text'].apply(lambda x: clean_text(x))
db['label'] = db['score'].map({0.0:0, 1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 6.0:1, 7.0:1, 8.0:1, 9.0:1, 10.0:1})
```
Diatas adalam proses Clearning proses Dimana menghapus karakter selain huruf seperti (@),(#) dan pemberian nilai pada scoree jika score 1- 5 bersifat 0(negatif), jika 6-10 bersifat 1(positif)<br>
b. memberi fitur tambahan
```python
def count_punct(text):
    if len(text) == 0 or (len(text) - text.count(" ")) == 0:
        return 0
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100
db['review_text_len'] = db['review_text'].apply(lambda x: len(x) - x.count(" "))
db['punct'] = db['review_text'].apply(lambda x: count_punct(x))
db.head()
```
Disini kita menambahkan pajang review tersebut dan seberapa sering tanda baca keluar pada review tersebut.<br>
c. Tokenization
```python
def tokenize_text(text):
    tokenized_text = text.split()
    return tokenized_text
db['perkata'] = db['cleaned_text'].apply(lambda x: tokenize_text(x))
db.head()
```
Disini kita mengubah kalimat pada review menjadi perkata.<br>
d. Lemmatization and removing Stopwords
```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
```
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
def lemmatize_text(token_list):
    return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])

lemmatizer = nltk.stem.WordNetLemmatizer()
db['lemmatized_review'] = db['perkata'].apply(lambda x: lemmatize_text(x))
db.head()
```
Disini kita mengubah kata makna menjadi kata biasa(atau kata yang tak bermakna) dan disini kita mengfiltering data frame.<br>
5.EDA
```python
print(f"Input data has {len(db)} rows and {len(db.columns)} columns")
print(f"score 0.0 = {len(db[db['score']==0.0])} rows")
print(f"score 1.0 = {len(db[db['score']==1.0])} rows")
print(f"score 2.0 = {len(db[db['score']==2.0])} rows")
print(f"score 3.0 = {len(db[db['score']==3.0])} rows")
print(f"score 4.0 = {len(db[db['score']==4.0])} rows")
print(f"score 5.0 = {len(db[db['score']==5.0])} rows")
print(f"score 6.0 = {len(db[db['score']==6.0])} rows")
print(f"score 7.0 = {len(db[db['score']==7.0])} rows")
print(f"score 8.0 = {len(db[db['score']==8.0])} rows")
print(f"score 9.0 = {len(db[db['score']==9.0])} rows")
print(f"score 10.0 = {len(db[db['score']==10.0])} rows")
```
Disini kita membentuk data frame dan membagikannya menjadi berbagai golongan.
```python
print(f"Number of null in label: { db['score'].isnull().sum() }")
print(f"Number of null in text: { db['review_text'].isnull().sum()}")
sns.countplot(x='score', data=db);
```
Kita mengecek review yang hanya emotikon atau tidak ada reviewnya.
 Dalam proses ini EDA ini dapat membedakan kumpulan data apa yang diungkapkan lebih jauh diluar pemodelan data formal ataupun data pengujian hipotesis.
 Dan disini saya memdeskripsikannya dalam diagram batang.<br>
 6. Visualizing Word Clouds<br>
 Word Clouds ini adalah suatu gambar yang terdiri dari kumpulan kata dimana besarnya kata mepresentasikan suatu kata yang sering keluar atau disebutkan dan akan ditampilkan pada dokumen text.
 ```python
from wordcloud import WordCloud
db_negative = db[ (db['score']==0.0) | (db['score']==1.0) | (db['score']==2.0)| (db['score']==3.0)| (db['score']==4.0)  | (db['score']==5.0)]
db_positive = db[ (db['score']==6.0) | (db['score']==7.0) | (db['score']==8.0) | (db['score']==9.0) | (db['score']==10.0)]
#convert to list
negative_list=db_negative['lemmatized_review'].tolist()
positive_list= db_positive['lemmatized_review'].tolist()

filtered_negative = ("").join(str(negative_list)) #convert the list into a string of spam
filtered_negative = filtered_negative.lower()

filtered_positive = ("").join(str(positive_list)) #convert the list into a string of ham
filtered_positive = filtered_positive.lower()
```
disini saya mengkategorikan score positif dan score negatif.<br>
a. Word Clouds :Positif
```python
wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Greens").generate(filtered_positive)
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("Positive Reviews Word Cloud")
plt.show()
```
b. Word Clouds : Negatif
```python
wordcloud = WordCloud(max_font_size = 160, margin=0, background_color = "white", colormap="Reds").generate(filtered_negative)
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("Negative Reviews Word Cloud")
plt.show()
```

7. Feature Extraction From Text
```python
X = db[['lemmatized_review', 'review_text_len', 'punct']]
y = db['label']
print(X.shape)
print(y.shape)
```
Extraksi fitur ini merupakan fitur reduksi yang mengubah dataset menjadi jumlah variabel yang lebih sedikit.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```
Diatas adalah model Sklearn
```python
from sklearn.feature_extraction.text import  TfidfVectorizer
tfidf =  TfidfVectorizer(max_df = 0.5, min_df = 2) # ignore terms that occur in more than 50% documents and the ones that occur in less than 2
tfidf_train = tfidf.fit_transform(X_train['lemmatized_review'])
tfidf_test = tfidf.transform(X_test['lemmatized_review'])

X_train_vect = pd.concat([X_train[['review_text_len', 'punct']].reset_index(drop=True), 
           pd.DataFrame(tfidf_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['review_text_len', 'punct']].reset_index(drop=True), 
           pd.DataFrame(tfidf_test.toarray())], axis=1)

X_train_vect.head()
X_test_vect.head() 
```
Proses diatas akan menghasilkan extraksi dari fitur diatas.<br>
8. Visualize Confusion Matrix<br>
--> Vectorizer : TF-IDF
   ```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
   ```
disini saya memanggil classification_report dan confusion_matrix<br>
-->Algoritma : Multinominal Naive Bayen
```python
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
#Mengganti nama kolom menjadi string
X_train_vect.columns = X_train_vect.columns.astype(str)
X_test_vect.columns = X_test_vect.columns.astype(str)


classifier.fit(X_train_vect, y_train)
naive_bayes_pred = classifier.predict(X_test_vect)

# Classification Report
print(classification_report(y_test, naive_bayes_pred))

# Confusion Matrix
class_label = ["negative", "positive"]
db_cm = pd.DataFrame(confusion_matrix(y_test, naive_bayes_pred), index=class_label, columns=class_label)
sns.heatmap(db_cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
```
Disini saya menggunakan algoritma multinominal Naive Bayen, dan proses diatas akan menghasilkan confusional Matrix adalah sebuah tabel yang sering digunakan untuk mengukur kinerja dari model klasifikasi di machine learning yang dapat membandingkan nilai aktual dari nilai asli dan nilai prediksi.<br>
9. Performing K-Fold cross Validation<br>
Merupakan sebuah prosedur untuk memisahkan data traning dan data testing. yang berfungsi untuk menemukan data yang terbaik.
```python
from sklearn.model_selection import cross_val_score

models = [MultinomialNB()]
names = ["Naive Bayes"]
for model, name in zip(models, names):
    print(name)
    for score in ["accuracy", "precision", "recall", "f1"]:
        print(f" {score} - {cross_val_score(model, X_train_vect, y_train, scoring=score, cv=10).mean()} ")
    print()
```
Diproses ini akan menghasilkan nilai Accurarcy, precision, recall dan f1.<br>

10. Prediction<br>
--> Vectorizer : CountVectorizer (Bag of Words)<br>
--> Algoritma : Multinominal Naive Bayen
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_cv = cv.fit_transform(db['lemmatized_review']) # Fit the Data
y_cv = db['label']

from sklearn.model_selection import train_test_split
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_cv, y_cv, test_size=0.3, random_state=42)
```
```python
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

clf.fit(X_train_cv, y_train_cv)
clf.score(X_test_cv, y_test_cv)
```
Didalam proses ini kita menemukan kesimpulan nilai dari classifier dalam review film tersebut adalah 0.8092234923137565
```python
data = ["Bad", "Good", "I hate the service, it's really bad", "The nurse is so kind"]
vect = cv.transform(data).toarray()

my_prediction = clf.predict(vect)
print(my_prediction)
```
Untuk Kesimpulan Menghasilakn [1.1.1.1] Mengartikan bahwa Sentimen analis kebanyakan review positif.
## Peforma Model
Klasifikasi Naive Bayen  memiliki kinerja yang baik pada masalah teks proccesing.
Tetapi jika kita menggunakan CountVectorizer akan hanya menghitung jumlah frekuensi kata tiap dokumenya saja jadi hasuilnya bilangan cacah atau bulat, jika dibuat untuk analisa kurang akurat karena tidak memperhatikan bobot tiap dokumennya. 
## Kesimpulan
Hasil dari pembuatan model Machine Learning yang dilakukan menghasilkan nilai prediksi 0.8092234923137565, 81% Keakuratan  dari total data 6764 untuk training dan 1691 untuk testing. Model trersebut menggunakan model sentimen analisis dan menggunakan algoritma naive Bayen. dan juga terdapat haris [1,1,1,1] dan dan dapat diartikan kebanyakan orang yang mengreview film tersebut memberi kesan positif pada film yang ditonton.

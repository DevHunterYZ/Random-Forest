# İris dataset ile kütüphanesini yükle.
from sklearn.datasets import load_iris

# Scikit'in rasgele orman sınıflandırıcı kütüphanesini yükleyin.
from sklearn.ensemble import RandomForestClassifier

# Pandas yükle.
import pandas as pd

# Numpy yükle.
import numpy as np

# Rastgele değer ayarla.
np.random.seed(0)
# İris verisiyle iris denilen bir nesne oluştur.
iris = load_iris()

# Dört özellik değişkeniyle bir veri çerçevesi oluştur.
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# İlk 5 satırı görüntüle.
df.head()
# Tür isimleri ile yeni bir sütun ekleyin, tahmin etmeye çalışacağız.
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# İlk 5 satırı görüntüle.
df.head()
# Her satır için 0 ile 1 arasında rasgele bir sayı oluşturan yeni bir sütun oluşturun ve bu değer 0,75'ten küçük veya ona eşitse, o hücrenin değerini True olarak ayarlar.
# Aksi takdirde False döndürür. Bu, bazı satırları rastgele bir şekilde atamanın hızlı ve kirli bir yoludur.
# Eğitim verileri ve bazıları test verileri olarak kullanılabilir.
df ['is_train'] = np.random.uniform (0, 1, len (df)) <= .75

# İlk 5 satırı görüntüle
df.head ()
# Biri test satırı olan diğeride bir eğitim(train) satırı olan iki yeni veri tablosu oluşturun.
train, test = df[df['is_train']==True], df[df['is_train']==False]
# Test ve eğitim veri çerçeveleri için gözlem sayısını göster.
print('Eğitim verilerindeki gözlem sayısı:', len(train))
print('Test verilerindeki gözlem sayısı:',len(test))

# Özellik sütununun adlarının bir listesini oluşturun.
features = df.columns[:4]

# Özellikleri görüntüle.
features
# train['species'] gerçek tür isimlerini içerir. Kullanmadan önce,
# Her türün adını bir basamağa dönüştürmemiz gerekiyor. Yani, bu durumda orada 0, 1 veya 2 olarak kodlanmış üç tür vardır.
y = pd.factorize(train['species'])[0]
# Hedefi görüntüle.
y
# Rastgele bir orman sınıflandırıcısı oluşturun. Sözlüğe göre, clf 'Sınıflandırıcı' anlamına gelir
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Eğitim özelliklerini almak ve ilişkilerini öğrenmek için sınıflandırıcıyı eğitin.
clf.fit(train[features], y)

# Test verisi için eğitilen Sınıflandırıcıyı uygulayın(daha önce hiç görmediğini hatırlayın).
clf.predict(test[features])

# İlk 10 gözlemin tahmin edilen olasılıklarını görüntüle.
clf.predict_proba(test[features])[0:10]

# Her tahmini bitki sınıfı için bitkiler için gerçek ingilizce isimleri oluştur.
preds = iris.target_names[clf.predict(test[features])]
# İlk beş gözlem için tahmin edilen türleri görüntüleyin.
preds[0:5]
# İlk beş gözlem için gerçek türleri görüntüleyin.
test['species'].head()
# Karışıklık matrisi oluştur.
pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
# Özelliklerin ve önem puanlarının bir listesini görüntüleyin.
list(zip(train[features], clf.feature_importances_))





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
# Her satır için 0 ile 1 arasında rasgele bir sayı oluşturan yeni bir sütun oluşturun ve
# bu değer 0,75'ten küçük veya ona eşitse, o hücrenin değerini True olarak ayarlar.
# aksi takdirde False döndürür. Bu, bazı satırları rastgele bir şekilde atamanın hızlı ve kirli bir yoludur.
# Eğitim verileri ve bazıları test verileri olarak kullanılabilir.
df ['is_train'] = np.random.uniform (0, 1, len (df)) <= .75

# En iyi 5 satırı görüntüle
df.head ()
# Biri test satırı olan diğeride bir eğitim(train) satırı olan iki yeni veri tablosu oluşturun.
train, test = df[df['is_train']==True], df[df['is_train']==False]
# Test ve eğitim veri çerçeveleri için gözlem sayısını göster.
print('Eğitim verilerindeki gözlem sayısı:', len(train))
print('Test verilerindeki gözlem sayısı:',len(test))


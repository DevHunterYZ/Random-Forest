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

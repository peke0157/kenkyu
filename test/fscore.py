import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path('testdata')
npy_files = list(data_dir.glob('**/*.npy'))

X = []
Y = []

for npy_file in npy_files:
    mfccs = np.load(npy_file)
    
    mean = np.mean(mfccs, axis=1)
    std = np.std(mfccs, axis=1)
    concatenate = np.concatenate([mean, std])
    
    genru = npy_file.parent.name
    
    X.append(concatenate)
    Y.append(genru)
    
X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)
print(np.unique(Y))    

print("Xのデータ数：", len(X))
print("Yのデータ数：", len(Y))

# テスト用データと訓練用データに分ける
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# SVMで学習させる
# 必要なモジュールをインポートする
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

clf = SVC()
clf.fit(x_train, y_train)

# 予測
y_pred = clf.predict(x_test)

# 結果の評価
print("混同行列")
print(confusion_matrix(y_test, y_pred))
print("分類レポート")
print(classification_report(y_test, y_pred))   




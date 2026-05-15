import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from huggingface_hub import HfApi

# ----------------------
# 1. 加载数据集
# ----------------------
df = pd.read_csv("imdb_balanced_10k.csv")
texts = df["review"].values
labels = df["sentiment"].values  # 0=负面 1=正面

# ----------------------
# 2. 文本向量化 TF-IDF
# ----------------------
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X = tfidf.fit_transform(texts).toarray()
y = labels

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------
# 3. 简单神经网络模型
# ----------------------
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2层神经网络
    activation="relu",
    max_iter=15,
    random_state=42,
    verbose=True
)

print("开始训练模型...")
model.fit(X_train, y_train)

# ----------------------
# 4. 评估
# ----------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {acc:.4f}")

# ----------------------
# 5. 保存模型 + 向量器
# ----------------------
joblib.dump(model, "model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
print("模型与TF-IDF向量化器已保存")

# ----------------------
# 6. 准备上传文件
# ----------------------
files_to_upload = [
    "model.joblib",
    "tfidf_vectorizer.joblib",
    "train.py",
    "requirements.txt",
    "imdb_balanced_10k.csv",
    "README.md"
]

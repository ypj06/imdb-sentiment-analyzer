import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from huggingface_hub import HfApi

# ----------------------
# 1. 加载 imdb_top_500 数据集
# ----------------------
df = pd.read_csv("imdb_top_500.csv")
print("✅ 数据集加载成功，列名：", df.columns.tolist())

# 读取文本和标签
texts = df["text"].values
labels = df["label"].values  # 0=负面 1=正面

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
    hidden_layer_sizes=(64, 32),  # 2层神经网络，符合任务要求
    activation="relu",
    max_iter=15,
    random_state=42,
    verbose=True
)

print("\n🚀 开始训练模型...")
model.fit(X_train, y_train)

# ----------------------
# 4. 模型评估
# ----------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 测试集准确率: {acc:.4f}")

# ----------------------
# 5. 保存模型 + 向量器
# ----------------------
joblib.dump(model, "model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
print("\n💾 模型与TF-IDF向量化器已保存")

# ----------------------
# 6. 上传文件列表（适配新数据集）
# ----------------------
files_to_upload = [
    "model.joblib",
    "tfidf_vectorizer.joblib",
    "train.py",
    "requirements.txt",
    "imdb_top_500.csv",
    "README.md"
]

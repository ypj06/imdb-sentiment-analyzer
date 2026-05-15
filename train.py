import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ----------------------
# 1. 加载数据集 imdb_top_500.csv
# ----------------------
df = pd.read_csv("imdb_top_500.csv")
print("数据集列名:", df.columns.tolist())

# 正确列名：text 和 label
texts = df["text"].values
labels = df["label"].values

# ----------------------
# 2. TF-IDF 文本向量化
# ----------------------
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X = tfidf.fit_transform(texts).toarray()
y = labels

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------
# 3. 神经网络模型
# ----------------------
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=20,
    random_state=42,
    verbose=True
)

print("\n开始训练...")
model.fit(X_train, y_train)

# ----------------------
# 4. 评估
# ----------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {acc:.4f}")

# ----------------------
# 5. 保存模型
# ----------------------
joblib.dump(model, "model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
print("模型保存完成！")

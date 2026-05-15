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
# 2. 优化版 TF-IDF（捕捉词组特征，抑制噪音）
# ----------------------
tfidf = TfidfVectorizer(
    max_features=8000,       # 扩大特征数量，捕捉更多文本细节
    stop_words="english",
    ngram_range=(1, 2),      # 加入二元词组，比如 "not good"
    sublinear_tf=True        # 降低高频词权重，提升低频关键特征的影响
)
X = tfidf.fit_transform(texts).toarray()
y = labels

# ----------------------
# 3. 分层抽样划分数据集（关键！保证正负样本分布均衡）
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# 4. 优化版神经网络（解决不收敛，提升泛化能力）
# ----------------------
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # 增加第一层神经元，提升表达能力
    activation="relu",
    max_iter=100,                  # 大幅增加迭代次数，解决不收敛问题
    early_stopping=True,           # 自动停止，防止过拟合
    n_iter_no_change=10,           # 连续10轮无提升则停止
    random_state=42,
    verbose=True
)

print("\n开始训练...")
model.fit(X_train, y_train)

# ----------------------
# 5. 评估
# ----------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {acc:.4f}")

# ----------------------
# 6. 保存模型
# ----------------------
joblib.dump(model, "model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
print("模型保存完成！")

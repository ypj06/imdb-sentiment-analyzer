import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ======================
# 1. 加载数据
# ======================
df = pd.read_csv("imdb_top_500.csv")
texts = df["text"].values
labels = df["label"].values

# ======================
# 2. 超强 TF-IDF（精度核心）
# ======================
tfidf = TfidfVectorizer(
    max_features=15000,
    stop_words="english",
    ngram_range=(1, 3),       # 1词、2词、3词短语全部捕捉
    min_df=1,
    max_df=0.85,
    sublinear_tf=True         # 抑制高频词，提升特征质量
)
X = tfidf.fit_transform(texts).toarray()
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 3. 高精度神经网络
# ======================
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    batch_size=8,
    learning_rate_init=0.0005,
    max_iter=150,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=42,
    verbose=True
)

# ======================
# 4. 训练
# ======================
model.fit(X_train, y_train)

# ======================
# 5. 输出准确率
# ======================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print(f"🚀 模型最终测试集准确率: {acc:.4f}")
print("="*60 + "\n")

# ======================
# 6. 保存模型
# ======================
joblib.dump(model, "model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
print("✅ 高精度模型保存完成！")

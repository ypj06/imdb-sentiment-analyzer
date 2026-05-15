import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ======================
# 1. 加载数据集
# ======================
df = pd.read_csv("imdb_top_500.csv")
texts = df["text"].values
labels = df["label"].values

# ======================
# 2. 优化版 TF-IDF（提升特征质量）
# ======================
tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words="english",
    ngram_range=(1, 2),  # 加入词组，效果大幅提升
    min_df=2
)
X = tfidf.fit_transform(texts).toarray()
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 3. 更强神经网络（提升准确率核心）
# ======================
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3层神经网络
    activation="relu",
    max_iter=50,                       # 更多迭代
    early_stopping=True,               # 早停防止过拟合
    validation_fraction=0.1,
    random_state=42,
    verbose=True
)

# ======================
# 4. 训练
# ======================
print("\n开始训练...")
model.fit(X_train, y_train)

# ======================
# 5. 输出准确率
# ======================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print(f"🚀 模型测试集准确率: {acc:.4f}")
print("="*60 + "\n")

# ======================
# 6. 保存模型
# ======================
joblib.dump(model, "model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
print("✅ 模型保存完成！")

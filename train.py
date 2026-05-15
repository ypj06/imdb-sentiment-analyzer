import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# ----------------------
# 1. 加载数据集
# ----------------------
df = pd.read_csv("imdb_top_500.csv")
texts = df["text"].values
labels = df["label"].values

# ----------------------
# 2. 分层抽样，保证正负样本分布均衡
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ----------------------
# 3. 文本向量化 + 分类器一体化 Pipeline
# ----------------------
pipeline = make_pipeline(
    TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    ),
    LogisticRegression(
        C=5.0,
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    )
)

# ----------------------
# 4. 训练
# ----------------------
print("\n开始训练...")
pipeline.fit(X_train, y_train)

# ----------------------
# 5. 评估
# ----------------------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print(f"🚀 测试集准确率: {acc:.4f}")
print("="*60 + "\n")

# ----------------------
# 6. 保存模型
# ----------------------
joblib.dump(pipeline, "model.joblib")
print("✅ 模型保存完成！")

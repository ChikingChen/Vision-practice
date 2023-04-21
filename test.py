from sklearn.model_selection import train_test_split

# 生成随机数据作为示例
import numpy as np
document_features = np.random.rand(1000, 10)
relevance_labels = np.random.randint(0, 3, size=(1000,))

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(document_features, relevance_labels, test_size=0.2, random_state=42)

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss="hinge", random_state=42)
sgd.fit(X_train, y_train)

# 对测试集的文档进行评分
relevance_scores = sgd.decision_function(X_test)

# 按照评分降序对文档索引进行排序
# 返回从小到大的索引序号
sorted_indices = np.argsort(-relevance_scores)

from sklearn.metrics import classification_report

y_true = y_test
y_pred = sgd.predict(X_test)
print(classification_report(y_true, y_pred))
import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture


def euclidean_distance(x1, x2):
    """
    计算两点之间的欧氏距离。
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=2):
        self.k = k

    def fit(self, X, y):
        """
        训练模型。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        对数据集X进行类别预测。
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        对单个样本进行类别预测。
        """
        # 计算距离
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # 获取k个最近样本的索引
        k_indices = np.argsort(distances)[: self.k]
        # 获取这些样本的类别
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 多数投票
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def test_knn():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=133
    )
    print(X_train, y_train)

    # 初始化KNN分类器
    clf = KNN(k=4)
    clf.fit(X_train, y_train)

    # 进行预测
    predictions = clf.predict(X_test)

    # 计算准确率
    print(f"KNN分类准确率: {accuracy_score(y_test, predictions)}")


def kmeans(X, k, init_centroids=None, max_iters=100):
    # 随机初始化簇心
    if init_centroids is None:
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices]
    else:
        centroids = init_centroids

    for _ in range(max_iters):
        # 初始化簇集合
        clusters = {i: [] for i in range(k)}

        # 分配步骤：为每个点分配最近的簇心
        for idx, x in enumerate(X):
            distances = np.linalg.norm(x - centroids, axis=1)
            closest = np.argmin(distances)
            clusters[closest].append(idx)

        # 更新步骤：计算每个簇的新簇心
        new_centroids = np.zeros(centroids.shape)
        for cluster_idx in clusters:
            new_centroid = np.mean(X[clusters[cluster_idx]], axis=0)
            new_centroids[cluster_idx] = new_centroid

        # 检查簇心是否发生变化
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    def predict_labels(X):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels.reshape(-1, 1), distances[np.arrange(len(X), labels)]

    return predict_labels, centroids


def gmm(X, k, inti_centroid=None, max_iters=100):
    gmm = GaussianMixture(n_components=k, random_state=0)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    return labels, probs, gmm.means_


def build_mask(factors, method="gmm", max_iters=20):
    factors = abs(factors)
    w = (factors - factors.min()) / (factors.max() - factors.min())
    if method == "gmm":
        print(f"Using classification method GMM")
        classify = gmm
    elif method == "kmeans":
        print(f"Using classification method KMeans")
        classify = kmeans
    else:
        raise ValueError("Method not found")

    labels, probs, centers = classify(w, k=2, max_iters=max_iters)

    # make sure that 0 represents the unmovable parts
    if abs(centers[0]) > abs(centers[1]):
        mask = np.ones_like(labels) - labels
        centers = centers[-2:]
    else:
        mask = labels

    return mask, centers


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    k = 4
    # centroids, clusters = kmeans(X, k)
    # print(clusters)

    labels, probs, centers = gmm(X, k)
    print(labels)
    size = 10 * probs.max(1) ** 2
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=size)
    plt.title("GMM Clustering")

    # 可视化簇心
    plt.scatter(centers[:, 0], centers[:, 1], c="red", s=300, alpha=0.6)
    plt.show()

    # 可视化结果
    # plt.figure(figsize=(12, 8))
    # for cluster_idx, cluster in clusters.items():
    #     cluster_points = X[cluster]
    #     plt.scatter(
    #         cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx}"
    #     )
    # plt.scatter(
    #     centroids[:, 0], centroids[:, 1], s=300, c="red", label="Centroids"
    # )  # 簇心
    # plt.title("K-Means Clustering")
    # plt.legend()
    # plt.show()

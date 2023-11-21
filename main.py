import numpy as np
import matplotlib.pyplot as plt
import torch

# Importing the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting data into training and testing set
def train_test_split(X, y, test_size=0.25, random_state=0):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_index = int((1 - test_size) * len(X))
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
def standard_scaler(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

# K-Nearest Neighbors model using PyTorch
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        distances = torch.norm(self.X_train - X_test, dim=1)
        _, indices = distances.topk(self.k, largest=False)
        k_nearest_labels = self.Y_train[indices]
        predicted_labels, _ = torch.mode(k_nearest_labels)
        return predicted_labels.numpy()

knn_classifier = KNNClassifier(k=5)
knn_classifier.fit(X_train_tensor, Y_train_tensor)

# Predicting for age=30, estimated salary=87000
new_data = torch.tensor([[30, 87000]], dtype=torch.float32)
predicted_label = knn_classifier.predict(new_data)
print(predicted_label)

# Predicting for test data
Y_pred_tensor = torch.tensor([knn_classifier.predict(x) for x in X_test_tensor], dtype=torch.float32).view(-1, 1)
Y_pred = Y_pred_tensor.numpy()
print(np.concatenate((Y_pred, Y_test.reshape(len(Y_pred), 1)), axis=1))

# Visualizing the Test set results
def plot_decision_boundary(X, Y, classifier, title):
    X_set, y_set = X.numpy(), Y.numpy().flatten()
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1),
                         np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1))
    plt.contourf(X1, X2, classifier.predict(torch.tensor(np.array([X1.ravel(), X2.ravel()]).T, dtype=torch.float32)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

plot_decision_boundary(X_test_tensor, Y_test_tensor, knn_classifier, 'K-NN (Test set)')

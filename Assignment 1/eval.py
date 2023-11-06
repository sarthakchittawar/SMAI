import numpy as np
import heapq
import sklearn.model_selection, sklearn.metrics, sklearn.neighbors
import random
import sys
from tabulate import tabulate

# Higher accuracy rate because of tie-breaking
class Best_KNN_Classifier:
    def __init__(self, encoder, k, distance_metric):
        if (encoder != 'resnet' and encoder != 'vit'):
            print("Encoder must be 'resnet' or 'vit'")
            return
        elif encoder == 'resnet':
            encoder = 1
        else:
            encoder = 2

        if (distance_metric != 'l1' and distance_metric != 'l2' and distance_metric != 'cosine'):
            print("Distance metric must be 'l1', 'l2' or 'cosine'")
            return
        elif distance_metric == 'l1':
            distance_metric = self.L1_distance
        elif distance_metric == 'l2':
            distance_metric = self.L2_distance
        else:
            distance_metric = self.cosine_similarity
            
        self.encoder = encoder
        self.k = k
        self.distance_metric = distance_metric

    # Calculates Manhattan Distance
    def L1_distance(self, x, y):
        return np.sum(np.abs(x - y), axis=1)

    # Calculates Euclidean Distance
    def L2_distance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y), axis=1))

    # Calculates 1 - Cosine Similarity
    def cosine_similarity(self, x, y):
        return 1 - (np.dot(y, x).astype(float) / (np.linalg.norm(x) * np.linalg.norm(y, axis=1)).astype(float)).astype(float)

    # Shows the given parameters
    def show(self):
        print(self.encoder, self.k, self.distance_metric)

    # Predicts the label of the given image data
    def pred(self, x, y, dist):
        k = self.k
        ksmallest = heapq.nsmallest(self.k, dist, key=lambda x: x[0])
        labels = {}
        for i in ksmallest:
            labels[y[int(i[1])]] = (labels[y[int(i[1])]] + 1) if y[int(i[1])] in labels else 1

        prediction = max(labels, key=labels.get)
        
        a = np.array([i for i in labels.items()])
        b = [i[0] for i in a if int(i[1]) == int(labels[prediction])]
        if (len(b) != 1):
            c = [[i[0] for i in ksmallest if y[int(i[1])] == j] for j in b]
            c = np.array(c)
            c = np.mean(c, axis=1) # can use mean or max for equivalent results
            smallest = heapq.nsmallest(1, c)
            c = list(c)
            ind = c.index(smallest)
            prediction = b[ind]
                
        return prediction

    # Calculates the distances of the given validation data from the training data
    def calculate_distances(self, X_train, X_valid):
        z = X_train[:, self.encoder]
        trainx = np.array([z[i][0] for i in range(len(z))])
        validx = X_valid[:, self.encoder][j][0]
        dist = np.concatenate((self.distance_metric(validx, trainx).reshape(-1, 1), np.arange(len(trainx), dtype=int).reshape(-1, 1)), axis=1)
        return dist

    # Calculates the accuracy, precision, recall and f1-score of the model on a given train-validation split
    def scores(self, X_train, y_train, X_valid, y_valid):
        y_pred = []
        z = X_train[:, self.encoder]
        trainx = np.array([z[i][0] for i in range(len(z))])
        
        for j in range(len(X_valid)):
            validx = X_valid[:, self.encoder][j][0]
            dist = np.concatenate((self.distance_metric(validx, trainx).reshape(-1, 1), np.arange(len(trainx), dtype=int).reshape(-1, 1)), axis=1)
            y_pred.append(self.pred(X_valid[j], y_train, dist))
        
        accuracy = sklearn.metrics.accuracy_score(y_valid, y_pred)
        precision = sklearn.metrics.precision_score(y_valid, y_pred, average='weighted', zero_division=np.nan)
        recall = sklearn.metrics.recall_score(y_valid, y_pred, average='weighted', zero_division=np.nan)
        f1_score = sklearn.metrics.f1_score(y_valid, y_pred, average='weighted')

        return accuracy, precision, recall, f1_score
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} <test_path.npy>".format(sys.argv[0]))
        sys.exit(1)

file_path = sys.argv[1]
data2 = np.load(file_path, allow_pickle=True)
data = np.load("data.npy", allow_pickle=True)

X_train, X_valid, y_train, y_valid = data, data2, np.array([i[3] for i in data]), np.array([i[3] for i in data2])
z = X_train[:, 2]
trainx = np.array([z[i][0] for i in range(len(z))])
z = X_valid[:, 2]
validx = np.array([z[i][0] for i in range(len(z))])

best_knn = Best_KNN_Classifier("vit", 7, "l2")
scores = best_knn.scores(X_train, y_train, X_valid, y_valid)

print("For a KNN Classifier with K = 7, using ViT as the encoder and L2 Distance as the distance metric:")
headers = ["Accuracy", "Precision", "Recall", "F1-Score"]
print(tabulate([scores], headers=headers))

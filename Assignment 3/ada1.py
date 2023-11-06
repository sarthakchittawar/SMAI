import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import wandb
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder

f = open('WineQT.csv', 'r')
headers = f.readline().strip().split(',')
data = []
x = f.readline()
while x != '':
    data.append([float(y) for y in x.strip().split(',')])
    x = f.readline()
    
data = np.array(data)
df = pd.DataFrame(data, columns=headers)

normalised_data = preprocessing.StandardScaler().fit_transform(data)
df = pd.DataFrame(normalised_data, columns=headers)

# dropping the quality and Id columns
quality = data[:, -2]
data = normalised_data
headers = headers[:-2]
data = data[:, :-2]

df = pd.DataFrame(data, columns=headers)

X_train, X_test, y_train, y_test = train_test_split(np.arange(data.shape[0]), np.arange(quality.shape[0]), test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
one_hot = pd.get_dummies(quality).values.astype(int)

one_hot_train = one_hot[X_train]
one_hot_val = one_hot[X_val]
one_hot_test = one_hot[X_test]

X_train = data[X_train]
X_val = data[X_val]
X_test = data[X_test]

y_train = quality[y_train]
y_val = quality[y_val]
y_test = quality[y_test]

class MultinomialLogisticRegression:
    def __init__(self, lr, max_epochs=10000, print_stats=False):
        self.learning_rate = lr
        self.max_epochs = max_epochs
        self.weights = None
        self.classes = None
        self.prev_loss = None
        self.print_stats = print_stats
    
    def softmax(self, x):
        # x is a 2D array
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
    
    def cross_entropy_gradient(self, X, z_softmax, one_hot):                    
        p = z_softmax - one_hot
        p = p.reshape(p.shape[0], p.shape[1], 1)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        grad = np.sum(p * X, axis=0)       
                    
        return grad
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.weights = np.random.randn(self.classes.shape[0], X.shape[1])
        self.bias = np.random.randn(self.classes.shape[0])
        epochs = 0
        
        # one hot encoding
        one_hot = pd.get_dummies(y).values.astype(int)
        
        train_losses = []
        train_accs = []
        val_accs = []
        
        while True:
            z = X @ self.weights.T + self.bias
            z_softmax = self.softmax(z)
                
            # calculate gradient
            weight_grad = self.cross_entropy_gradient(X, z_softmax, one_hot)
            bias_grad = np.sum(z_softmax - one_hot, axis=0)
                        
            # update weights
            self.weights -= self.learning_rate * weight_grad
            self.bias -= self.learning_rate * bias_grad
            
            # calculate loss            
            p = z_softmax @ one_hot.T
            loss = np.sum(-np.log(np.diag(p) / np.sum(z_softmax, axis=1)))
            
            
            epochs += 1
            if self.print_stats and (epochs % 100 == 0 or epochs == 1):
                print('Epoch: ', epochs, ' Loss: ', loss)
                        
                train_losses.append(loss)
                pred = self.predict(X)
                accuracy = metrics.accuracy_score(y, pred)
                train_accs.append(accuracy)
                
                pred = self.predict(X_val)
                accuracy = metrics.accuracy_score(y_val, pred)
                val_accs.append(accuracy)
            
            if self.prev_loss is not None and (np.abs(self.prev_loss - loss) < 1e-5 or epochs > self.max_epochs):
                self.prev_loss = loss
                break
            self.prev_loss = loss
            
        if self.print_stats:
            plt.plot(range(len(train_losses)), train_losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
            
            plt.plot(range(len(train_accs)), train_accs, label='Training Accuracy')
            plt.plot(range(len(val_accs)), val_accs, label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
                
        return self.weights, self.bias
    
    def predict(self, X):
        z = X @ self.weights.T + self.bias
        z_softmax = self.softmax(z)
                        
        y_pred = np.argmax(z_softmax, axis=1)
        return self.classes[y_pred]

# hyperparameter tuning
print('Hyperparameter tuning started')

learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
epochs = 20000

sweep_configuration = {
    "name": "logistic-regression-sweep",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "method": "grid",
    "parameters": {"lr": {"values": learning_rates}}
}

def my_train_func():
    wandb.init()
    lr = wandb.config.lr
    
    model = MultinomialLogisticRegression(lr, epochs)
    w, b = model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    val_accuracy = metrics.accuracy_score(y_val, y_pred)
    
    
    for i in range(100, epochs+1, 100):
        wandb.log({"lr": lr, "val_accuracy": val_accuracy, "epoch": i})

sweep_id = wandb.sweep(sweep_configuration, project='logistic-regression')

wandb.agent(sweep_id, function=my_train_func)
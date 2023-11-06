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

df = pd.read_csv('advertisement.csv')
X = df.drop('labels', axis=1)
y = df.iloc[:, -1].values
# display(X)

for i in range(len(y)):
    y[i] = np.array(y[i].split(' '))
y = np.array(y)

s = X.shape[1]
# print(LabelEncoder().fit_transform(X.values[:, 3]))
X['gender'] = LabelEncoder().fit_transform(X.values[:, 1])
X['education'] = LabelEncoder().fit_transform(X.values[:, 3])
X['married'] = LabelEncoder().fit_transform(X.values[:, 4])
X['occupation'] = LabelEncoder().fit_transform(X.values[:, 7])
X['most bought item'] = LabelEncoder().fit_transform(X.values[:, 9])

df = X.drop(['city'], axis=1)
X = df.values.astype(float)
# display(df)

X = preprocessing.StandardScaler().fit_transform(X)
df = pd.DataFrame(X)
# display(df)

# Splitting data into train, validation and test sets

X_train, X_test, y_train, y_test = train_test_split(np.arange(X.shape[0]), np.arange(y.shape[0]), test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
classes = []
for i in y:
    for j in i:
        if j not in classes:
            classes.append(j)
            
classes = np.array(classes)
one_hot = MultiLabelBinarizer(classes=classes).fit_transform(y)

one_hot_train = one_hot[X_train]
one_hot_val = one_hot[X_val]
one_hot_test = one_hot[X_test]

X_train = X[X_train]
X_val = X[X_val]
X_test = X[X_test]

y_train = y[y_train]
y_val = y[y_val]
y_test = y[y_test]

class MultiLabel_MLPClassifier:
    def __init__(self, lr, activation_func, optimizer, hidden_layer_sizes, max_epochs=2000, print_stats=False, batch_size=100):
        self.learning_rate = lr
        self.activation_func = activation_func
        self.optimizer = optimizer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layers = len(hidden_layer_sizes)
        self.print_stats = print_stats
        self.max_epochs = max_epochs
        self.prev_loss = None
        self.weights = None
        self.biases = None
        self.weights_grad = None
        self.biases_grad = None
        self.batch_size = batch_size
        
    # def softmax(self, x):
    #     # x is a 2D array
    #     return np.exp(x) / (np.sum(np.exp(x), axis=1).reshape(-1, 1) + 1e-15)
    
    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.where(x > 0, x, 0)
    
    def tanh(self, x, derivative=False):
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)
    
    def forward_propagation(self, X):
        z = []
        # input layer
        z.append(X)
        x = np.matmul(X, self.weights[0]) + self.biases[0]
        next_x = self.activation_funcs[0](x)
                    
        # hidden layers
        for i in range(1, self.hidden_layers):
            z.append(next_x)
            x = np.matmul(next_x, self.weights[i]) + self.biases[i]
            next_x = self.activation_funcs[i](x)
            
        # output layer
        z.append(next_x)
        x = np.matmul(next_x, self.weights[-1]) + self.biases[-1]
        final = self.sigmoid(x)
        
        return z, final
    
    def backpropagation(self, final, z, one_hot):
        # output layer
        p = final - one_hot
        op_grad = p
        
        # hidden layers
        for i in range(self.hidden_layers, -1, -1):
            dw = z[i].reshape(z[i].shape[0], z[i].shape[1], 1)
            op = op_grad.reshape(op_grad.shape[0], 1, op_grad.shape[1])
            self.weights_grad[i] = np.sum(dw * op, axis=0)
            self.biases_grad[i] = np.sum(op_grad, axis=0)
            if i == 0:
                break
            dz = np.matmul(op_grad, self.weights[i].T) / op_grad.shape[0]
            
            dz = dz * self.activation_funcs[i-1](z[i], derivative=True)
            
            op_grad = dz
        
        return self.weights_grad, self.biases_grad
        
    def fit(self, X, y, X_val, y_val):
        
        if self.activation_func == 'sigmoid':
            self.activation_func = self.sigmoid
        elif self.activation_func == 'relu':
            self.activation_func = self.relu
        elif self.activation_func == 'tanh':
            self.activation_func = self.tanh
        else:
            raise Exception('Invalid activation function')
        
        if self.optimizer == 'sgd':
            self.batch_size = 1
        elif self.optimizer == 'batch':
            self.batch_size = X.shape[0]
        elif self.optimizer == 'mini-batch':
            self.batch_size = self.batch_size
            if self.batch_size > X.shape[0] or self.batch_size < 1 or X.shape[0] % self.batch_size != 0:
                raise Exception('Invalid batch size')
        else:
            raise Exception('Invalid optimizer')
        
        # can non-linearize from outside
        self.activation_funcs = []
        for i in range(self.hidden_layers):
            self.activation_funcs.append(self.activation_func)                
        
        classes = []
        for i in y:
            for j in i:
                if j not in classes:
                    classes.append(j)

        self.classes = np.unique(classes)
        self.weights = []
        self.biases = []
        
        # one_hot = pd.get_dummies(y).values.astype(int)
        one_hot = MultiLabelBinarizer(classes=self.classes).fit_transform(y)
        one_hot_val = MultiLabelBinarizer(classes=self.classes).fit_transform(y_val)
        
        weights = []
        biases = []
        weights_grad = []
        biases_grad = []
                
        # input layer
        weights.append(np.random.randn(X.shape[1], self.hidden_layer_sizes[0]))
        biases.append(np.zeros((self.hidden_layer_sizes[0])))
        weights_grad.append(np.zeros((X.shape[1], self.hidden_layer_sizes[0])))
        biases_grad.append(np.zeros((self.hidden_layer_sizes[0])))
        
        # hidden layers until last hidden layer
        for i in range(self.hidden_layers - 1):
            weights.append(np.random.randn(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1]))
            biases.append(np.zeros((self.hidden_layer_sizes[i+1])))
            weights_grad.append(np.zeros((self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1])))
            biases_grad.append(np.zeros((self.hidden_layer_sizes[i+1])))
            
        # last hidden layer
        weights.append(np.random.randn(self.hidden_layer_sizes[-1], self.classes.shape[0]))
        biases.append(np.zeros((self.classes.shape[0])))
        weights_grad.append(np.zeros((self.hidden_layer_sizes[-1], self.classes.shape[0])))
        biases_grad.append(np.zeros((self.classes.shape[0])))
        
        losses = []
        
        self.weights = weights
        self.biases = biases
        self.weights_grad = weights_grad
        self.biases_grad = biases_grad
        
        full_X = X
        one_hot_full = one_hot
        
        epochs = 0
        val_accuracies = []
        train_accuracies = []
        while True:
            # shuffle data
            p = np.random.permutation(full_X.shape[0])
            X = full_X[p]
            one_hot = one_hot_full[p]
            
            X = X.reshape(-1, self.batch_size, X.shape[1])
            one_hot = one_hot.reshape(-1, self.batch_size, one_hot.shape[1])            
            
            sum = 0
            for i in range(X.shape[0]):
                # forward propagation
                z, final = self.forward_propagation(X[i])
                final = np.clip(final, 1e-15, 1)
                
                # backpropagation
                self.backpropagation(final, z, one_hot[i])
                    
                # update weights
                for j in range(self.hidden_layers + 1):
                    self.weights[j] -= self.learning_rate * self.weights_grad[j]
                    self.biases[j] -= self.learning_rate * self.biases_grad[j]
                                
                # calculate loss
                p = one_hot[i] * np.log(final) + (1 - one_hot[i]) * np.log(1 - final)
                loss = -np.mean(p)
                sum += loss
            sum /= X.shape[0]
            losses.append(sum)
            loss = sum
            
            _, y_pred, _ = self.predict(full_X)
            # hl = 0
            # for i in range(len(y)):
            #     hl += metrics.hamming_loss(one_hot_full[i], y_pred[i])
            accuracy = 1 - metrics.hamming_loss(one_hot_full, y_pred)
            train_accuracies.append(accuracy)
            _, y_pred, _ = self.predict(X_val)
            # hl = 0
            # for i in range(len(y_val)):
            #     hl += metrics.hamming_loss(one_hot_val[i], y_pred[i])
            accuracy = 1 - metrics.hamming_loss(one_hot_val, y_pred)
            val_accuracies.append(accuracy)
                
            epochs += 1
            if self.print_stats and (epochs % 100 == 0 or epochs == 1):
                print('Epoch: ', epochs, 'Loss: ', loss, 'Train Accuracy: ', train_accuracies[-1], 'Val Accuracy: ', val_accuracies[-1])
                
            if epochs > self.max_epochs or (self.prev_loss is not None and np.abs(self.prev_loss - loss) < 1e-5):
                self.prev_loss = loss
                break
        if self.print_stats:    
            plt.plot(range(len(losses)), losses)
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()              
            
        return weights, biases, train_accuracies, val_accuracies
            
    def predict(self, X):
        _, final = self.forward_propagation(X)
        y_pred = np.where(final > 0.5, 1, 0)
        labels = [self.classes[np.where(row == 1)[0]] for row in y_pred]
        
        return final, y_pred, labels
    
    
# hyperparameter tuning
print('Hyperparameter tuning started')
# wandb.init(project="mlp-classifier")

learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
epochs = 2000
hidden_layer_sizes = [[6, 6], [8, 8], [6, 8], [6, 6, 6], [8,8,8], [6, 8, 10]]
optimizers = ['sgd', 'batch', 'mini-batch']
activation_funcs = ['sigmoid', 'tanh', 'relu']

sweep_configuration = {
    "name": "multilabel-mlp-classification-sweep",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "method": "grid",
    "parameters": {"lr": {"values": learning_rates},
                   "activation_func": {"values": activation_funcs},
                   "optimizer": {"values": optimizers},
                   "hidden_layers": {"values": hidden_layer_sizes}}
}

def my_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init()
    lr = wandb.config.lr
    activation_func = wandb.config.activation_func
    optimizer = wandb.config.optimizer
    hidden_layers = wandb.config.hidden_layers
    
    mlp = MultiLabel_MLPClassifier(lr, activation_func, optimizer, hidden_layers, epochs)
    
    _, _, train_acc, val_acc = mlp.fit(X_train, y_train, X_val, y_val)
    final, y_pred, _ = mlp.predict(X_val)
    # precision = metrics.precision_score(y_val, y_pred, average='macro', zero_division=np.nan)
    # recall = metrics.recall_score(y_val, y_pred, average='macro', zero_division=np.nan)
    # f1 = metrics.f1_score(y_val, y_pred, average='macro', zero_division=np.nan)
    
    conf_matrix = metrics.multilabel_confusion_matrix(one_hot_val, y_pred)
            
    tp, fp, tn, fn = 0, 0, 0, 0
    f1 = []

    for i in conf_matrix:
        tn += i[0][0]
        tp += i[1][1]
        fn += i[0][1]
        fp += i[1][0]
        if ((i[1][1] + i[1][0]) == 0 or (i[1][1] + i[0][1]) == 0 or i[1][1] == 0):
            continue

        precision = i[1][1] / (i[1][1] + i[1][0])
        recall = i[1][1] / (i[1][1] + i[0][1])
        f1.append(2 * precision * recall / (precision + recall))

    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f1_micro = (2 * precision * recall) / (precision + recall)
    f1_macro = np.mean(f1)

    val_loss = -np.mean(one_hot_val * np.log(final) + (1 - one_hot_val) * np.log(1 - final))
    final, _, _ = mlp.predict(X_train)
    train_loss = -np.mean(one_hot_train * np.log(final) + (1 - one_hot_train) * np.log(1 - final))
    for i in range(100, epochs+1, 100):
        wandb.log({"learning_rate": lr, "activation_func": activation_func, "optimizer": optimizer, "epochs": i, "hidden_layers": hidden_layers, "val_accuracy": val_acc[i-1], "train_accuracy": train_acc[i-1], "final_precision": precision, "final_recall": recall, "final_f1-score-micro": f1_micro, "final_f1-score-macro": f1_macro, "val_loss": val_loss, "train_loss": train_loss})


sweep_id = wandb.sweep(sweep_configuration, project='multilabel-mlp-classifier')

wandb.agent(sweep_id, function=my_train_func)
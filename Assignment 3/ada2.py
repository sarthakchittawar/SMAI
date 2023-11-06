import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import wandb
from tqdm import tqdm
from sklearn.impute import SimpleImputer

f = open('HousingData.csv', 'r')
headers = f.readline().strip().split(',')
data = []
x = f.readline()
while x != '':
    data.append([y for y in x.strip().split(',')])
    for i in range(len(data[-1])):
        if data[-1][i] == 'NA':
            data[-1][i] = np.nan
    x = f.readline()
# data = f.read()
data = np.array(data)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)

df = pd.DataFrame(data, columns=headers)

normalised_data = preprocessing.StandardScaler().fit_transform(data)
medv = data[:, -1]
data = normalised_data
headers = headers[:-1]
data = data[:, :-1]

df = pd.DataFrame(data, columns=headers)

# Splitting data into train, validation and test sets

X_train, X_test, y_train, y_test = train_test_split(np.arange(data.shape[0]), np.arange(medv.shape[0]), test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

X_train = data[X_train]
X_val = data[X_val]
X_test = data[X_test]

y_train = medv[y_train]
y_val = medv[y_val]
y_test = medv[y_test]

class MLPRegression:
    def __init__(self, lr, activation_func, optimizer, hidden_layer_sizes, max_epochs=2000, print_stats=False, batch_size=3):
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
        final = np.matmul(next_x, self.weights[-1]) + self.biases[-1]
        
        return z, final
    
    def backpropagation(self, final, z, y):
        # output layer
        # print("backprop")
        # p = final - y
        op_grad = np.diag(2 * (final - y)).reshape(-1, 1)
        
        # hidden layers
        for i in range(self.hidden_layers, -1, -1):
            # print(z[i].shape, op_grad.shape)
            dw = z[i].reshape(z[i].shape[0], z[i].shape[1], 1)
            op = op_grad.reshape(op_grad.shape[0], 1, op_grad.shape[1])
            # print(dw.shape, op.shape)
            # print(dw)
            # print(op)
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
        
        self.weights = []
        self.biases = []
                
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
        weights.append(np.random.randn(self.hidden_layer_sizes[-1], 1))
        biases.append(np.zeros((1)))
        weights_grad.append(np.zeros((self.hidden_layer_sizes[-1], 1)))
        biases_grad.append(np.zeros((1)))
        
        losses = []
        
        self.weights = weights
        self.biases = biases
        self.weights_grad = weights_grad
        self.biases_grad = biases_grad
        
        full_X = X
        full_y = y
        
        epochs = 0
        val_mse = []
        train_mse = []
        train_r2 = []
        val_r2 = []
        
        while True:
            # shuffle data
            p = np.random.permutation(full_X.shape[0])
            X = full_X[p]
            y = full_y[p]
            
            X = X.reshape(-1, self.batch_size, X.shape[1])
            y = y.reshape(-1, self.batch_size)            
            
            sum = 0
            for i in range(X.shape[0]):
                # forward propagation
                z, final = self.forward_propagation(X[i])
                
                # backpropagation
                self.backpropagation(final, z, y[i])
                    
                # update weights
                for j in range(self.hidden_layers + 1):
                    self.weights[j] -= self.learning_rate * self.weights_grad[j]
                    self.biases[j] -= self.learning_rate * self.biases_grad[j]
                                
                # calculate loss
                p = metrics.mean_squared_error(y[i], final)
                loss = np.mean(p)
                sum += loss
            sum /= X.shape[0]
            losses.append(sum)
            loss = sum
            
            y_pred = self.predict(full_X)
            train_mse.append(metrics.mean_squared_error(full_y, y_pred))
            train_r2.append(metrics.r2_score(full_y, y_pred))
            # accuracy = metrics.accuracy_score(y, y_pred)
            # train_accuracies.append(accuracy)
            y_pred = self.predict(X_val)
            val_mse.append(metrics.mean_squared_error(y_val, y_pred))
            val_r2.append(metrics.r2_score(y_val, y_pred))
            # accuracy = metrics.accuracy_score(y_val, y_pred)
            # val_accuracies.append(accuracy)
                
            epochs += 1
            if self.print_stats and (epochs % 100 == 0 or epochs == 1):
                print('Epoch: ', epochs, 'Loss: ', loss, 'Train MSE: ', train_mse[-1], 'Val MSE: ', val_mse[-1], 'Train R2: ', train_r2, 'Val R2: ', val_r2)
                
            if epochs > self.max_epochs or (self.prev_loss is not None and np.abs(self.prev_loss - loss) < 1e-5):
                self.prev_loss = loss
                break
        if self.print_stats:    
            plt.plot(range(len(losses)), losses)
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()              
            
        return weights, biases, train_mse, val_mse, train_r2, val_r2
            
    def predict(self, X):        
        z, final = self.forward_propagation(X)
        return final
    
sweep_configuration = {
    "name": "mlp-regression-sweep",
    "metric": {"name": "val_mse", "goal": "minimize"},
    "method": "grid",
    "parameters": {"lr": {"values": [1e-3, 1e-4, 1e-5]},
                   "activation_func": {"values": ['sigmoid', 'relu', 'tanh']},
                   "optimizer": {"values": ['sgd', 'batch', 'mini-batch']},
                   "hidden_layers": {"values": [[6, 6], [7, 7], [8, 8], [6, 6, 6], [8,8,8]]}}
}

def my_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init()
    lr = wandb.config.lr
    activation_func = wandb.config.activation_func
    optimizer = wandb.config.optimizer
    hidden_layers = wandb.config.hidden_layers
    
    mlp = MLPRegression(lr, activation_func, optimizer, hidden_layers, 500)
    _, _, train_mse, val_mse, train_r2, val_r2 = mlp.fit(X_train, y_train, X_val, y_val)
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
        
    for i in range(50, 501, 50):
        wandb.log({"train_mse": train_mse[i], "val_mse": val_mse[i], "train_rmse": train_rmse[i], "val_rmse": val_rmse[i], "train_r2": train_r2[i], "val_r2": val_r2[i], "epochs": i, "lr": lr, "activation_func": activation_func, "optimizer": optimizer, "hidden_layers": hidden_layers})


sweep_id = wandb.sweep(sweep_configuration, project='mlp-regression')

# run the sweep
wandb.agent(sweep_id, function=my_train_func)
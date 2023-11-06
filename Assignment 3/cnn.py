import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
data = torchvision.datasets.MNIST('./mnist', train=True, transform=transform)
# split this data into training and validation sets
train, val = train_test_split(data, test_size=0.2)
test = torchvision.datasets.MNIST('./mnist', train=False, transform=transform)

print("Training set size:", len(train))
print("Validation set size:", len(val))
print("Test set size:", len(test))

data2 = data
data = train

imgs = []
labels = []
for i in data:
    imgs.append(torchvision.transforms.ToPILImage()(i[0]))
    labels.append(i[1])

# imgs = np.array(imgs)
labels = np.array(labels)

for i in range(10):
    labels2 = labels - i
    print("Count of", i, "=", len(labels) - np.count_nonzero(labels2))

x = np.arange(10)

plt.hist(labels, rwidth=0.8)
plt.title('Labels v/s Frequency')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks(np.arange(10), [str(i) for i in range(10)])
plt.show()

count = np.zeros((10)).astype(int)

l = [[0 for i in range(5)] for j in range(10)]

for i in data:
    if (count[i[1]] < 5):
        l[i[1]][count[i[1]]] = i[0][0]
        count[i[1]] += 1
        
# print("First 5 images of each label:")
# for i in range(10):
#     for j in range(5):
#         plt.subplot(5, 10, i + j * 10 + 1)
#         plt.imshow(l[i][j])
#         plt.axis('off')
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'

d = {}
d[2, 1] = 36864
d[2, 2] = 2304
d[2, 3] = 576
d[3, 1] = 25600
d[3, 2] = 1024
d[3, 3] = 256
d[4, 1] = 16384
d[4, 2] = 576
d[4, 3] = 64

class CNN_model(torch.nn.Module):
    def __init__(self, kernel_size=3, dropout_rate=0.5, stride=3, out1 = 32, out2 = 64):
        super(CNN_model, self).__init__()
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.out1 = out1
        self.out2 = out2

        # first layer
        self.conv1 = torch.nn.Conv2d(1, self.out1, self.kernel_size, padding=0)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=0)

        # second layer
        self.conv2 = torch.nn.Conv2d(self.out1, self.out2, self.kernel_size, padding=0)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=0)

        size = d[self.kernel_size, self.stride] * self.out2 / 64
        size = int(size)
        # third layer
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.layer3 = torch.nn.Linear(size, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, X):
        X = self.maxpool1(self.relu1(self.conv1(X)))
        X = self.maxpool2(self.relu2(self.conv2(X)))
        X = X.reshape(X.shape[0],-1)
        X = self.layer3(self.dropout(X))
        X = self.softmax(X)

        return X

# wrapper class for CNN_model
class CNN:
    def __init__(self, learning_rate=0.01, num_epochs=10, batch_size=32, kernel_size=3, dropout_rate=0.5, stride=3, out1 = 32, out2 = 64):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.out1 = out1
        self.out2 = out2

    def fit(self, data, test):
        train_init, val = train_test_split(data, test_size=0.2)

        train = DataLoader(train_init, batch_size=self.batch_size, shuffle=True)
        val = DataLoader(val)
        test = DataLoader(test)

        print("Training set size:", len(train)*train.batch_size)
        print("Validation set size:", len(val))
        print("Test set size:", len(test))

        model = CNN_model(self.kernel_size, self.dropout_rate, self.stride, self.out1, self.out2).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train()

        epoch = 0

        val_accuracies = []
        train_accuracies = []
        val_losses = []
        train_losses = []

        while epoch < self.num_epochs:
            train = DataLoader(train_init, batch_size=self.batch_size, shuffle=True)
            for i, (X, y) in enumerate(train):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                X.detach()
                y.detach()

            epoch += 1

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                count = 0
                loss = 0
                for i, (X, y) in enumerate(val):
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    loss += criterion(y_pred, y).item()
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    X.detach()
                    y.detach()
                    y_pred.detach()
                    count += 1
                val_loss = loss / count
                val_acc = correct / total
                val_precision = metrics.precision_score(y.cpu(), predicted.cpu(), average='macro')
                val_recall = metrics.recall_score(y.cpu(), predicted.cpu(), average='macro')
                val_f1 = metrics.f1_score(y.cpu(), predicted.cpu(), average='macro')
                conf_matrix = metrics.confusion_matrix(y.cpu(), predicted.cpu())

                loss = 0
                count = 0
                for i, (X, y) in enumerate(train):
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    loss += criterion(y_pred, y).item()
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    X.detach()
                    y.detach()
                    y_pred.detach()
                    count += 1
                train_loss = loss / count
                train_acc = correct / total

                print('Epoch: {}, Train Loss: {}, Val loss: {}, Train Accuracy: {}, Val Accuracy: {}'.format(epoch, train_loss, val_loss, train_acc, val_acc))
                val_accuracies.append(val_acc)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                train_losses.append(train_loss)
            model.train()

        # model.detach()

        return train_accuracies, val_accuracies, train_losses, val_losses, val_precision, val_recall, val_f1, conf_matrix

# device = 'cpu'       
print(device)
# cnn = CNN(batch_size=len(data), learning_rate=1e-3, dropout_rate=0.25, num_epochs=10, kernel_size=3, stride=3)
# cnn.fit(data2, test)

# hyperparameter tuning
print('Hyperparameter tuning started')
# wandb.init(project="mlp-classifier")

learning_rates = [1e-2, 1e-3, 1e-4]
epochs = 20
dropout_rates = [0, 0.25, 0.5, 0.75]
kernel_sizes = [2, 3]
strides = [2, 3]
batch_sizes = [len(data)/1000, len(data)/100, len(data)/10]
out1s = [32, 64]
out2s = [32, 64]

sweep_configuration = {
    "name": "cnn-sweep",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "method": "random",
    "parameters": {"learning_rate": {"values": learning_rates},
                   "dropout_rate": {"values": dropout_rates},
                   "kernel_size": {"values": kernel_sizes},
                   "stride": {"values": strides},
                   "batch_size": {"values": batch_sizes},
                   "out1": {"values": out1s},
                   "out2": {"values": out2s}}
}

def my_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init()
    learning_rate = wandb.config.learning_rate
    dropout_rate = wandb.config.dropout_rate
    kernel_size = wandb.config.kernel_size
    stride = wandb.config.stride
    batch_size = wandb.config.batch_size
    out1 = wandb.config.out1
    out2 = wandb.config.out2
    epochs = 20
    
    if (kernel_size, stride) in d.keys():
    
        cnn = CNN(batch_size=batch_size, learning_rate=learning_rate, dropout_rate=dropout_rate, num_epochs=epochs, kernel_size=kernel_size, stride=stride, out1=out1, out2=out2)
        train_accuracies, val_accuracies, train_losses, val_losses, val_precision, val_recall, val_f1, conf_matrix = cnn.fit(data2, test)

        for i in range(epochs):
            wandb.log({'learning_rate': learning_rate, 'dropout_rate': dropout_rate, 'kernel_size': kernel_size, 'stride': stride, 'batch_size': batch_size, 'out1': out1, 'out2': out2, 'epoch': i + 1, 'train_accuracy': train_accuracies[i], 'val_accuracy': val_accuracies[i], 'train_loss': train_losses[i], 'val_loss': val_losses[i], 'val_precision': val_precision, 'val_recall': val_recall, 'val_f1': val_f1, 'conf_matrix': conf_matrix})

sweep_id = wandb.sweep(sweep_configuration, project='cnn')

wandb.agent(sweep_id, function=my_train_func)

# cnn = CNN(batch_size=48, learning_rate=1e-3, dropout_rate=0.25, num_epochs=20, kernel_size=3, stride=1)
# cnn.fit(data2, test)

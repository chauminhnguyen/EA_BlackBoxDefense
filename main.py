from ges import GES
import numpy as np
import torch
from cnn import Data, Net
import torch.nn as nn

data = Data()
net = Net()
criterion = nn.CrossEntropyLoss() 

def func(W):

    new_net = Net()
    new_net.set_weights(W)
    # new_net.load_state_dict(net.state_dict())
    index = np.random.randint(0, len(data.trainset))
    X, label = data.trainset[index]
    X = X.unsqueeze(0)
    label = torch.tensor(label).unsqueeze(0)
    outputs = new_net(X)
    loss = criterion(outputs, label)
    running_loss = loss.item()

    # running_loss = 0.
    # # return X**2 + X - 6
    # for i, ele in enumerate(data.trainloader):
    #     # get the inputs; data is a list of [inputs, labels]
    #     inputs, labels = ele
    #     outputs = net(inputs)
    #     loss = criterion(outputs, labels)
    #     running_loss += loss.item()
    return running_loss

def evaluate(W):
    new_net = Net()
    new_net.set_weights(W)
    running_loss = 0.
    for X, label in data.trainloader:
        X = X.unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        outputs = net(X)
        loss = criterion(outputs, label)
        running_loss += loss.item()
    return running_loss

# X = np.random.rand(1, 100) # batch, dim
# Y = ges(X, T=600, P=100, n=100, k=50, f=func, std=0.1, alpha=0.5, beta=2, eta=1e-7)
# Y = ges(X, T=749, P=100, n=100, k=32, f=func, std=1.0, alpha=0.9, beta=1.5, eta=0.005154351000000001)
if __name__ == '__main__':
    # Y = func()
    X = data.trainloader
    ges = GES(T=5, P=100, n=62006, k=500, f=func, update_X=net.set_weights, std=0.1, alpha=0.5, beta=2, eta=1e-7)
    Y = ges.run(net.get_weights())
    print(evaluate(Y))
# print(np.square(np.subtract(Y,Y_)).mean())

# def objective(trial):
#     X = np.random.rand(1, 100) # batch, dim

#     T = trial.suggest_int('T', 2, 1000)
#     k = trial.suggest_int('k', 10, 100)
#     std = trial.suggest_float('std', 0.1, 1., step=0.1)
#     alpha = trial.suggest_float('alpha', 0.1, 1., step=0.1)
#     beta = trial.suggest_float('beta', 0.1, 10., step=0.1)
#     eta = trial.suggest_float('eta', 1e-9, 1e-2, step=1e-8)
    
#     Y = ges(X, T=T, P=100, n=100, k=k, f=func, std=std, alpha=alpha, beta=beta, eta=eta)
#     Y_ = func(X)
#     return np.square(np.subtract(Y,Y_)).mean()

# study = optuna.create_study()
# study.optimize(objective, n_trials=500)

# print(study.best_params)  # E.g. {'x': 2.002108042}
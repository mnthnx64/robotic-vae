
from torch import nn
import sklearn.datasets
import torch
import numpy as np
from torch import optim

class Classifier():
# Layer details for the neural network
    def __init__(self, input_dim, output_dim):
        self.input_size = input_dim
        self.hidden_sizes = [128, 64]
        self.output_size = output_dim

        # Build a feed-forward network
        self.model = nn.Sequential(nn.Linear(self.input_size, self.hidden_sizes[0]),
                              nn.ReLU(),
                              nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                              nn.ReLU(),
                              nn.Linear(self.hidden_sizes[1], self.output_size),
                              nn.LogSoftmax(dim=1))

        self.criterion = nn.functional.cross_entropy

        # Optimizers require the parameters to optimize and a learning rate
        # self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)

        # time0 = time()
        self.epochs = 100


        self.running_loss = 0

        self.accuracy = 0

    def train(self, X, y):
        for e in range(self.epochs):
            # Flatten MNIST images into a 784 long vector
            # images = images.view(images.shape[0], -1)

            # Training pass
            self.optimizer.zero_grad()
            
            output = self.model(X)

            loss = self.criterion(output, y)
            
            #This is where the model learns by backpropagating
            loss.backward(retain_graph=True)
            
            #And optimizes its weights here
            self.optimizer.step()
            
            self.running_loss += loss.item()


        print(self.running_loss)

    def test(self, X, y):
        predict_y = self.model(X)
        # print(predict_y)
        # print(predict_y[0])
        total_count = 0
        correct_count = 0
        
        for val in predict_y:

            prob_tensor = torch.exp(val)

            temp = prob_tensor.detach().numpy()

            prob = list(temp)

            # print(prob)
            
            pred_label = prob.index(max(prob))

            # print(pred_label)

            
            if(pred_label == y[total_count]):
                correct_count += 1
            total_count += 1

        print("Accuracy: ")
        print(correct_count / total_count)
        self.accuracy += (correct_count / total_count)
                
        # print(y[0])


 
# X, y = sklearn.datasets.make_classification(n_samples=100, n_features=13, n_classes = 5, n_redundant=0, random_state=42, n_informative=13)

# print(X[0])
# print(y[0])

# X = torch.from_numpy(X).type(torch.FloatTensor)
# y = torch.from_numpy(y).type(torch.LongTensor)
# X_test = X[0:50]
# y_test = y[0:50]
# X = X[51:100]
# y = y[51:100]
# nn = Classifier(13,5)
# nn.train(X, y)
# nn.test(X_test , y_test)



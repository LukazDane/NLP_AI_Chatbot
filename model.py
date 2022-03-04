import torch
import torch.nn as nn
import torch.functional as F


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Initialize our layers and their inputs/outputs
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        # ReLu Activation function

    def forward(self, x):
        # First Layer takes in x
        out = self.l1(x)
        out = self.relu(out)
        # Second layer, takes out (above from first layer) as input, and outputs a new update of out
        out = self.l2(out)
        out = self.relu(out)
        # Third layer
        out = self.l3(out)
        # No activation function here
        return out


class AdvancedNeuralNet(nn.Module):
    def __init__(self, bert, input_size, hidden_size, num_classes):
        super(AdvancedNeuralNet, self).__init__()
        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()
        # dense layer
        self.l1 = nn.Linear(768, input_size)
        self.l2 = nn.Linear(input_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
        # define the forward pass

    def forward(self, sent_id, mask):
        # pass the inputs to the model
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]

        x = self.l1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.l3(x)

        # apply softmax activation
        x = self.softmax(x)
        return x

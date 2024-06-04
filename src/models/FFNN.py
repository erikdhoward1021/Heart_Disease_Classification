import torch
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.2):
        super(FFNN, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.last = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out = self.input(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.last(out)
        out = self.sigmoid(out)
        return out
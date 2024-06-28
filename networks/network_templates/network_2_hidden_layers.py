from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, f_active_1, f_active_2, f_active_3):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.f_active_1 = f_active_1
        self.f_active_2 = f_active_2
        self.f_active_3 = f_active_3


    def forward(self, x):
        out = self.fc1(x)
        out = self.f_active_1(out)
        out = self.fc2(out)
        out = self.f_active_2(out)
        out = self.fc3(out)
        out = self.f_active_3(out)
        return out
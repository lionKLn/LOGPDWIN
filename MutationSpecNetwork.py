import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SingleFCLayer(nn.Module):
    def __init__(self, in_features, out_features, keep_prob):
        super(SingleFCLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=1 - keep_prob)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MutationSpecNetwork(nn.Module):
    def __init__(self, model_size_times, keep_prob):
        super(MutationSpecNetwork, self).__init__()
        model_size_times = 2

        # Separate mutation layers
        self.mut1 = SingleFCLayer(35, 35 * model_size_times, keep_prob)
        self.mut2 = SingleFCLayer(35, 35 * model_size_times, keep_prob)
        self.mut3 = SingleFCLayer(35, 35 * model_size_times, keep_prob)
        self.mut4 = SingleFCLayer(35, 35 * model_size_times, keep_prob)
        self.mut_concat = SingleFCLayer(35 * 4 * model_size_times, 35 * model_size_times, keep_prob)

        # Separate spec layers
        self.spec1 = SingleFCLayer(34, 34 * model_size_times, keep_prob)
        self.spec_concat_fc = SingleFCLayer(69 * model_size_times, 32 * model_size_times, keep_prob)

        # FC layers
        self.complex_fc = SingleFCLayer(37, 37 * model_size_times, keep_prob)
        self.similar_fc = SingleFCLayer(15, 15 * model_size_times, keep_prob)
        self.fc1 = SingleFCLayer(84 * model_size_times, 128, keep_prob)

        # Output layer
        self.final_weight = nn.Parameter(torch.randn(128, 2))
        self.final_bias = nn.Parameter(torch.zeros(2))

    def forward(self, m1, m2, m3, m4, spec, complexity, similarity):
        mut1_out = self.mut1(m1)
        mut2_out = self.mut2(m2)
        mut3_out = self.mut3(m3)
        mut4_out = self.mut4(m4)
        mut_concat_out = self.mut_concat(torch.cat([mut1_out, mut2_out, mut3_out, mut4_out], dim=1))

        spec1_out = self.spec1(spec)
        spec_concat = self.spec_concat_fc(torch.cat([spec1_out, mut_concat_out], dim=1))

        complex_out = self.complex_fc(complexity)
        similar_out = self.similar_fc(similarity)
        fc1_out = self.fc1(torch.cat([spec_concat, complex_out, similar_out], dim=1))

        out_layer = torch.matmul(fc1_out, self.final_weight) + self.final_bias
        return out_layer


def main():
    # Initialize the network
    model = MutationSpecNetwork(model_size_times=2, keep_prob=0.5)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # or any other loss function suitable for your task
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data loading
    # Replace the following with your actual data loading mechanism
    train_loader = DataLoader(TensorDataset(...), batch_size=64, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # L2 Regularization
            loss += l2_regularization(model, lambda_l2=0.0003)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print or log the loss value if needed

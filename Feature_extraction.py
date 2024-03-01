from torch.nn import LSTM
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch.nn.functional import one_hot
import torch.nn.functional as F
import os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import time

def process_data(data):
    x_columns = data.columns[:-1]
    x = data[x_columns].values
    x = torch.FloatTensor(x)
    y = data['insider'].values
    y = torch.LongTensor(y)
    return x, y

class LstmAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, encode_size):
        super(LstmAutoEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.encode_size = encode_size

        self.encoder_lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.encoder_fc = nn.Linear(self.hidden_size, self.encode_size)
        self.decoder_lstm = nn.LSTM(self.encode_size, self.hidden_size, batch_first=True)  # Adjusted input_size
        self.decoder_fc = nn.Linear(self.hidden_size, self.input_size)  # Adjusted input_size
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        encoder_lstm, (n, c) = self.encoder_lstm(x)
        encoder_fc = self.encoder_fc(encoder_lstm)
        encoder_out = self.relu(encoder_fc)

        # decoder
        decoder_lstm, _ = self.decoder_lstm(encoder_out)  # Decoder input is the encoder output
        decoder_fc = self.decoder_fc(decoder_lstm)

        return decoder_fc

if __name__ == '__main__':
    file_path = 'ExtractedData/dayr4.2.csv'
    data = pd.read_csv(file_path)
    print(data.shape)
    print(data.head())

    # Define parameters
    epochs = 1000
    seq_len = len(data.columns)-1  # for full length sequential feature extraction
    lr = 0.01

    # Prepare data
    x, y = process_data(data)
    dataset = Data.TensorDataset(x, y)
    dataloader = Data.DataLoader(dataset, batch_size=2000, shuffle=True)

    # Instantiate model
    model = LstmAutoEncoder(input_size=seq_len, hidden_size=100, batch_size=2000, encode_size=5) # LSTM auto-encoder feature extraction output dimension

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}')

    # After training, you can extract temporal features using the encoder part of the model
    model.eval()
    with torch.no_grad():
        encoded_features = []
        for batch_x, _ in dataloader:
            encoded = model.encoder_lstm(batch_x)[0]  # Only taking the encoder output
            encoded = encoded.mean(dim=1)  # Pooling over time steps
            encoded_features.append(encoded)
        encoded_features = torch.cat(encoded_features, dim=0)

    # Now, `encoded_features` contains the temporal features extracted by the LSTM autoencoder
    encoded_features_df = pd.DataFrame(encoded_features.numpy())
    encoded_features_df.to_csv('feature_extraction.csv', index=False)
    
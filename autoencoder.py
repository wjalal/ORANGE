import pandas as pd
import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, compressed_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, input_dim),
            nn.Sigmoid()  # Use sigmoid for binary data, or remove for continuous data
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


gene_sort_crit = sys.argv[1]
if gene_sort_crit != '20p' and gene_sort_crit != '1000' and gene_sort_crit != 'deg' and gene_sort_crit != 'AA':
    print ("Invalid args")
    exit (1)
    
# organ_list = ["artery_coronary", "muscle_skeletal", "whole_blood", "skin_sun_exposed_lower_leg", "lung", "liver", "heart_left_ventricle", "nerve_tibial", "artery_aorta", "colon_transverse", "colon_sigmoid"]
with open('gtex/organ_list.dat', 'r') as file:
    organ_list = [line.strip() for line in file]

md_hot = pd.read_csv(filepath_or_buffer="../../../gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS-rangemid_int.txt", sep='\s+').set_index("SUBJID")

for organ in organ_list:
    print(organ)
    df_gene = pd.read_csv(filepath_or_buffer="../../../gtex/proc/proc_data/" + organ + ".tsv", sep='\s+').set_index("Name")
    # print (df_gene)
    df_gene.index.names = ['SUBJID']
    md_hot_organ = md_hot.merge(right = df_gene.index.to_series(), how='inner', left_index=True, right_index=True)
    # print(md_hot_organ)

    # Dataset Preparation (assuming `X_train` is your dataset)
    # Convert your data to PyTorch tensors
    scaler = PowerTransformer(method='yeo-johnson')
    tmp = scaler.fit_transform(df_gene)
    pd.DataFrame(tmp, index=df_gene.index, columns=df_gene.columns)
    X_train = torch.tensor(df_gene.values, dtype=torch.float32)
    train_dataset = TensorDataset(X_train, X_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Define hyperparameters
    input_dim = X_train.shape[1]  # Number of features
    compressed_dim = 1024  # Dimension after compression (choose based on how many features you want)
    learning_rate = 0.00001
    num_epochs = 10

    # Initialize model, loss function, and optimizer
    model = Autoencoder(input_dim=input_dim, compressed_dim=compressed_dim).to(device)
    criterion = nn.MSELoss()  # Reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for data, _ in train_loader:
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # After training, use the encoder part for feature selection
    def extract_features(model, data):
        model.eval()
        with torch.no_grad():
            encoded_data = model.encoder(data.to(device))
        return encoded_data.cpu()

    # Extract compressed features (selected features)
    X_compressed = extract_features(model, X_train)
    print(X_compressed.shape)  # (400, compressed_dim), e.g., (400, 64)

    df_gene.index.names = ['Name']
    # Convert the compressed features back to a Pandas DataFrame
    compressed_df = pd.DataFrame(X_compressed.numpy(), index=df_gene.index)

    # Save the DataFrame as a CSV file
    compressed_df.to_csv("../../../gtex/proc/proc_data/reduced/corr" + gene_sort_crit + "/" + organ + ".tsv", sep='\t', index=True)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import cv2
import os
from torchvision import models
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd




device = torch.device("cpu")


resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet.fc = torch.nn.Identity()  
resnet.eval()
resnet.to(device)


def extract_features(image_path, size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image / 255.0, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        features = resnet(image).squeeze().detach().cpu().numpy().flatten()
        
    return features.reshape(-1, 1) 


def create_graph(features, label, k=5):
    x = torch.tensor(features, dtype=torch.float).view(1, -1)  

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(x)
    edges = nbrs.kneighbors_graph(x).tocoo()
    edge_index = torch.tensor(np.vstack((edges.row, edges.col)), dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)
    
class APTOSGraphDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.graphs = []
        for _, row in self.labels.iterrows():
            img_path = os.path.join(img_dir, row["id_code"] + ".png")
            features = extract_features(img_path)
            graph = create_graph(features, row["diagnosis"], k=1)
            self.graphs.append(graph)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    





class DRGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(DRGNN, self).__init__()
        self.conv1 = GINConv(torch.nn.Linear(in_channels, hidden_channels))
        self.conv2 = GINConv(torch.nn.Linear(hidden_channels, hidden_channels))
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Predict(model, img_path):

    features = extract_features(img_path)  
    graph = create_graph(features, label=0, k=1)  
    
    graph = graph.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(graph)
        pred = torch.argmax(output, dim=1).item()
    return pred

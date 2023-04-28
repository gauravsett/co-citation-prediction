import torch
import torch.nn as nn
from torch.nn import ReLU, Dropout
from torch_geometric.nn import SAGEConv 
import torch.nn.functional as F
from regression import RegressionModel

class GraphModel(RegressionModel):
    
    def __init__(self, data, hidden_channels, dropout_p):
        super(GraphModel, self).__init__(data, hidden_channels)
        self.data = data
        self.num_features = data.features.size()[1]
        self.conv1 = SAGEConv(self.num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.act = ReLU()
        self.dropout = Dropout(p=dropout_p)


    def forward(self, data):
        
        out1 = self.act(self.conv1(data.features, data.edge_index))
        out1 = self.dropout(out1)

        out2 = self.conv2(out1)
        output = F.log_softmax(out2, dim=1)

        return output

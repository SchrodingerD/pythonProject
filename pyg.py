# import torch
# from torch_geometric.data import Data
#
# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index.t().contiguous())
# print(Data)

# from torch_geometric.datasets import TUDataset
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# print(len(dataset))

from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora',name='Cora')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features,16)
        self.conv2 = GCNConv(16,dataset.num_classes)
    def forward(self,data):
        x,edge_index = data.x, data.edge_index

        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x,training=self.training)
        x = self.conv2(x,edge_index)
        return F.log_softmax(x,dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)




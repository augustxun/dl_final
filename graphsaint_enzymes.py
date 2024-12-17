import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import GraphSAINTRandomWalkSampler, GraphSAINTSampler, GraphSAINTNodeSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, GATConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.data import Batch

dataset_name = 'ENZYMES' # 'PROTEINS'

if dataset_name == 'ENZYMES':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES')
    dataset = TUDataset(path, name='ENZYMES')
else:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PROTEINS')
    dataset = TUDataset(path, name='PROTEINS')

dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#train_data = Batch.from_data_list(train_dataset)
#train_loader = GraphSAINTSampler(train_data, batch_size=20)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GATConv(train_dataset.num_features, 64, heads=4, residual=True)
        self.pool1 = TopKPooling(4 * 64, ratio=0.8)
        self.conv2 = GATConv(4 * 64, 64, heads=4, residual=True)
        self.pool2 = TopKPooling(4 * 64, ratio=0.8)
        self.conv3 = GATConv(4 * 64, 64, heads=4, residual=True)
        self.pool3 = TopKPooling(4 * 64, ratio=0.8)

        self.lin1 = torch.nn.Linear(8 * 64, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(output.shape, data.y.shape)
        exit(0)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    total_pred = torch.LongTensor([])
    total_y = torch.LongTensor([])
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        #print(pred.shape, data.y.shape)
        total_pred = torch.cat((total_pred, pred.cpu()), 0)
        total_y = torch.cat((total_y, data.y.cpu()), 0)
        correct += pred.eq(data.y).sum().item()
    f1_res = f1_score(total_y, total_pred, average='macro')
    return f1_res, correct / len(loader.dataset)


best_val_f1 = -999
test_f1, test_acc = -999, -999
for epoch in range(1, 201):
    loss = train(epoch)
    val_f1, val_acc = test(val_loader)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        test_f1, test_acc = test(test_loader)
    #test_f1, test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Val F1: {val_f1:.5f}, Val Acc: {val_acc:.5f}, '
          f'Test F1: {test_f1:.5f}, Test Acc: {test_acc:.5f}')
print(f'Best Test F1: {test_f1:.4f}, Best Test Acc: {test_acc:.4f}')
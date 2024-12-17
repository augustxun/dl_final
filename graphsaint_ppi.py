import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv, GATConv
from torch_geometric.typing import WITH_TORCH_SPARSE
from torch_geometric.utils import degree
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import time

from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

train_data = Batch.from_data_list(train_dataset)
train_loader = GraphSAINTRandomWalkSampler(train_data, batch_size=1, walk_length=20,
                                     num_steps=5, sample_coverage=100,
                                     save_dir=train_dataset.processed_dir,
                                     num_workers=0, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = GraphConv(train_dataset.num_features, 64)
        #self.conv2 = GraphConv(64, 64)
        #self.conv3 = GraphConv(64, train_dataset.num_classes)
        self.conv1 = GATConv(train_dataset.num_features, 64, heads=4,
                             residual=True)
        self.conv2 = GATConv(4 * 64, 64, heads=4, residual=True)
        self.conv3 = GATConv(4 * 64, train_dataset.num_classes, heads=4,
                             concat=False, residual=True)
        #self.lin = torch.nn.Linear(3 * 64, train_dataset.num_classes)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index):
        x1 = F.relu(self.conv1(x0, edge_index))
        #x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index))
        #x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = self.conv3(x2, edge_index)
        #x3 = F.dropout(x3, p=0.2, training=self.training)
        #x = torch.cat([x1, x2, x3], dim=-1)
        #x = self.lin(x)
        #return x.log_softmax(dim=-1)
        return x3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_op = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    #model.set_aggr('add' if args.use_normalization else 'mean')
    #model.set_aggr('mean')

    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        #if args.use_normalization:
        #    edge_weight = data.edge_norm * data.edge_weight
        #    out = model(data.x, data.edge_index, edge_weight)
        #    loss = F.nll_loss(out, data.y, reduction='none')
        #    loss = (loss * data.node_norm)[data.train_mask].sum()
        #else:
        out = model(data.x, data.edge_index)
        #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        #print(out, data.y)
        #print(out.shape, data.y.shape)
        #exit(0)
        loss = loss_op(out, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    #model.set_aggr('mean')

    ys, preds = [], []
    outs = []
    for data in loader:
        ys.append(data.y)
        out = torch.sigmoid(model(data.x.to(device), data.edge_index.to(device)))
        outs.append(out.float().cpu())
        preds.append((out > 0.5).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    outs = torch.cat(outs, dim=0).numpy()
    f1_res = f1_score(y, pred, average='macro') if pred.sum() > 0 else 0
    acc_res = accuracy_score(y, pred) if pred.sum() > 0 else 0
    auc_res = roc_auc_score(y, outs, average='macro')
    return f1_res, acc_res, auc_res

    #out = model(data.x.to(device), data.edge_index.to(device))
    #pred = out.argmax(dim=-1)
    #correct = pred.eq(data.y.to(device))

    #accs = []
    #for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #   accs.append(correct[mask].sum().item() / mask.sum().item())
    #return accs


#for epoch in range(1, 51):
#    loss = train()
#    accs = test()
#    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
#          f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')


times = []
best_val_f1 = -999
test_f1, test_acc, test_auc = -999, -999, -999
for epoch in range(1, 201):
    start = time.time()
    loss = train()
    val_f1, val_acc, val_auc = test(val_loader)
    if val_f1 > best_val_f1:
        best_val_f1 = test_f1
        test_f1, test_acc, test_auc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f},'
          f'Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
print(f'Best Test F1: {test_f1:.4f}, Best Test Acc: {test_acc:.4f}, Best Test AUC: {test_auc:.4f}')


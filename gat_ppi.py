import os.path as osp
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(train_dataset.num_features, 64, heads=4,
                             residual=True)
        self.conv2 = GATConv(4 * 64, 64, heads=4, residual=True)
        self.conv3 = GATConv(4 * 64, train_dataset.num_classes, heads=4,
                             concat=False, residual=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    outs = []
    for data in loader:
        ys.append(data.y)
        out = torch.sigmoid(model(data.x.to(device), data.edge_index.to(device)))
        outs.append(out.cpu())
        preds.append((out > 0.5).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    outs = torch.cat(outs, dim=0).numpy()
    f1_res = f1_score(y, pred, average='macro') if pred.sum() > 0 else 0
    acc_res = accuracy_score(y, pred) if pred.sum() > 0 else 0
    auc_res = roc_auc_score(y, outs, average='macro')
    return f1_res, acc_res, auc_res


times = []
best_val_f1 = -999
test_f1, test_acc, test_auc = -999, -999, -999
for epoch in range(1, 201):
    start = time.time()
    loss = train()
    val_f1, val_acc, val_auc = test(val_loader)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        test_f1, test_acc, test_auc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f},'
          f'Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
print(f'Best Test F1: {test_f1:.4f}, Best Test Acc: {test_acc:.4f}, Best Test AUC: {test_auc:.4f}')

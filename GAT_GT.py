import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, GATConv, TransformerConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import time

dataset_name = 'ENZYMES' # 'PROTEINS'
#dataset_name = 'PROTEINS'

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
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gat_conv1 = GATConv(train_dataset.num_features, 64, heads=4, residual=True)
        self.gat_pool1 = TopKPooling(4 * 64, ratio=0.8)
        self.gat_conv2 = GATConv(4 * 64, 64, heads=4, residual=True)
        self.gat_pool2 = TopKPooling(4 * 64, ratio=0.8)
        self.gat_conv3 = GATConv(4 * 64, 64, heads=4, residual=True)
        self.gat_pool3 = TopKPooling(4 * 64, ratio=0.8)

        self.gt_conv1 = TransformerConv(train_dataset.num_features, 64, heads=4)
        self.gt_pool1 = TopKPooling(4 * 64, ratio=0.8)
        self.gt_conv2 = TransformerConv(4 * 64, 64, heads=4)
        self.gt_pool2 = TopKPooling(4 * 64, ratio=0.8)
        self.gt_conv3 = TransformerConv(4 * 64, 64, heads=4)
        self.gt_pool3 = TopKPooling(4 * 64, ratio=0.8)

        self.lin1 = torch.nn.Linear(2 * 8 * 64, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def gat_mp(self, x, edge_index, batch):
        x = F.relu(self.gat_conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.gat_pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.gat_conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.gat_pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.gat_conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.gat_pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        return x1 + x2 + x3

    def gt_mp(self, x, edge_index, batch):
        x = F.relu(self.gt_conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.gt_pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.gt_conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.gt_pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.gt_conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.gt_pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        return x1 + x2 + x3

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = self.gat_mp(x, edge_index, batch)
        x2 = self.gt_mp(x, edge_index, batch)
        x = torch.cat([x1, x2], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
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
    total_pred_source = torch.LongTensor([])
    total_pred_value = torch.LongTensor([])
    total_y_bin = torch.LongTensor([])
    for data in loader:
        data = data.to(device)
        pred_source = model(data)
        pred_value, pred = pred_source.max(dim=1)
        #print(pred.shape, data.y.shape)

        if dataset_name == 'ENZYMES':
            y_binarized = label_binarize(data.y, classes=[i for i in range(dataset.num_classes)])
            total_y_bin = torch.cat((total_y_bin, torch.tensor(y_binarized)), 0)

        total_pred_source = torch.cat((total_pred_source, pred_source.cpu()), 0)
        total_pred = torch.cat((total_pred, pred.cpu()), 0)
        total_pred_value = torch.cat((total_pred_value, pred_value.cpu()), 0)
        total_y = torch.cat((total_y, data.y.cpu()), 0)
        correct += pred.eq(data.y).sum().item()

    #print(total_y.shape, total_pred.shape, total_pred_source.shape)
    f1_res = f1_score(total_y, total_pred, average='macro')
    if dataset_name == 'ENZYMES':
        roc_auc_macro = roc_auc_score(total_y_bin.detach().numpy(), total_pred_source.detach().numpy(), multi_class='ovr', average='macro')
    else:
        roc_auc_macro = roc_auc_score(total_y, total_pred_value.detach().numpy(), multi_class='ovr', average='macro')
    return f1_res, correct / len(loader.dataset), roc_auc_macro


times = []
best_val_f1 = -999
test_f1, test_acc, test_auc = -999, -999, -999
for epoch in range(1, 201):
    start = time.time()
    loss = train(epoch)
    val_f1, val_acc, val_auc = test(val_loader)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        test_f1, test_acc, test_auc = test(test_loader)
    #test_f1, test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f},'
          f'Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
print(f'Best Test F1: {test_f1:.4f}, Best Test Acc: {test_acc:.4f}, Best Test AUC: {test_auc:.4f}')

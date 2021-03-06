import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from torch_geometric.data import download_url
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import time

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--dataset', default='Cora',
                    help='Define the dataset.')
parser.add_argument('--runmode', default='train',
                    help='Define the runmode (train/test)')
parser.add_argument('--niter', default=100,
                    help='Define the iteration number in training.')
parser.add_argument('--device', default='cpu',
                    help='Define the device.')
parser.add_argument('--hsize', default=16,
                    help='Define the hidden representation vector length.')
args = parser.parse_args()

# obtain the proper dataset
dataset_name=args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
if args.dataset.lower()=='reddit':
    dataset = Reddit(path)
else:
    dataset = Planetoid(path, dataset_name, T.NormalizeFeatures())
data = dataset[0]

model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models', dataset_name+'-'+str(args.hsize)+'.pth')

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

hsize=int(args.hsize)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hsize, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hsize, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # 0 dropout in Nell dataset
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

if args.device=='cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.runmode == 'train':
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=1e-5),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=0.0005)
else:
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    data = data.to(device)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    # time measurement
    if args.device == 'cpu':
        t_start = time.perf_counter()
    else:
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    # FIXME: test 100 iteration for a precise measurement
    if args.runmode == 'test' and args.dataset != 'Reddit':
        for ii in range(0, 100):
            logits, accs = model(), []
    else:
        logits, accs = model(), []
    if args.device == 'cpu':
        exe_time = (time.perf_counter() - t_start)*1000
    else:
        gpu_end.record()
        torch.cuda.synchronize()
        exe_time = gpu_start.elapsed_time(gpu_end)

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, exe_time


best_val_acc = test_acc = 0

if args.runmode == 'train':
    for epoch in range(1, int(args.niter)):
        train()
        [train_acc, val_acc, tmp_test_acc], exe_time = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    # save the PyTorch model
    torch.save(model.state_dict(), model_path)
    print("Model saved.")
else:
    [train_acc, val_acc, test_acc], exe_time = test()
    log = 'Test: {:.4f}'
    # print(log.format(test_acc))
    print("Time (ms):" + str(round(exe_time, 2)))

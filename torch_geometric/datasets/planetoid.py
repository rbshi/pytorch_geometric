import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data


class Planetoid(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    url_nell = 'https://github.com/rbshi/public/raw/master/nell_dataset'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(Planetoid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        # --rbshi
        if self.name.lower() == 'nell':
            return ['ind.{}.0.001.{}'.format(self.name.lower(), name) for name in names]
        else:
            return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            # --rbshi
            if self.name.lower() == 'nell':
                download_url('{}/{}'.format(self.url_nell, name), self.raw_dir)
            else:
                download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

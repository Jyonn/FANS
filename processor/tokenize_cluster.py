import json
import os

import numpy as np
from UniTok import UniDep, UniTok, Vocab


class ClusterTokenizer:
    def __init__(self, data_dir, cluster_json):
        self.depot = UniDep(data_dir)
        self.cluster_dict = json.load(open(cluster_json, 'r'))

    def tokenize(self):
        item_vocab = self.depot.vocab_depot.get_vocab('item_id')

        clusters = []
        for i in range(item_vocab.get_size()):
            clusters.append(int(self.cluster_dict[item_vocab.index2obj[i]]))

        self.store_dir = self.depot.store_dir + '-cls'
        os.makedirs(self.store_dir, exist_ok=True)

        data = dict(
            item_id=np.array(range(item_vocab.get_size()), dtype=object),
            cluster_id=np.array(clusters, dtype=object),
        )
        np.save(os.path.join(self.store_dir, 'data.npy'), data, allow_pickle=True)

        vocab_info = dict(
            item_id=dict(size=item_vocab.get_size(), cols=['item_id']),
            cluster_id=dict(size=max(clusters) + 1, cols=['cluster_id']),
        )

        col_info = dict(
            item_id=dict(vocab='item_id'),
            cluster_id=dict(vocab='cluster_id'),
        )

        meta_data = dict(
            version=UniTok.VER,
            vocab_info=vocab_info,
            col_info=col_info,
            id_col='item_id',
        )
        json.dump(meta_data, open(os.path.join(self.store_dir, 'meta.data.json'), 'w'))

        item_vocab.save(self.store_dir)
        cluster_vocab = Vocab(name='cluster_id')
        for i in range(max(clusters) + 1):
            cluster_vocab.append(i)
        cluster_vocab.save(self.store_dir)

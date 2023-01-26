# Datasets

Datasets are available here: [Google Drive](https://drive.google.com/drive/folders/1yAtYr__cm-VyzIZ0EVtVoRtlvN6wwm9U?usp=sharing)

## Dataset Format

Each dataset is a folder containing the following files:

* clusters folder: contains the clusters of the items in the dataset. Each cluster is a file containing the items in the cluster.
* data.npy: contains the data matrix of the dataset.
* meta.data.json: contains the metadata of the dataset.
* tok.cluster_id.dat: cluster id vocabulary.
* tok.global_id.dat: global item id vocabulary.
* tok.local_id.dat: local item id (i.e., item order in each cluster) vocabulary.
* tok.index.dat: index vocabulary.

## Dataset Statistics

|                        |  Zhihu   |  Spotify  |  AotM   | Goodreads |
|-----------------------:|:--------:|:---------:|:-------:|:---------:|
|                # Lists |  18,704  |  72,152   | 12,940  |  15,426   |
|                # Items |  36,005  |  104,695  |  6,264  |  47,877   | 
|             # Clusters |    20    |    20     |   20    |    20     |
|         # Interactions | 927,781  | 6,809,820 | 162,106 | 1,589,480 |
|  Avg. # items per list |  49.59   |   94.38   |  12.53  |  103.04   |
| Range # items per list | 10 ~ 200 | 20 ~ 300  | 10 ~ 60 | 20 ~ 300  |
|                Density |  0.138%  |  0.089%   | 0.200%  |  0.215%   |

## Dataset Examples

```python
from UniTok import UniDep
depot = UniDep('data/zhihu-n10')
print(depot[0]) 
```

```json
{
  "index": 0,
  "k_global": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 
  "p_global": [494, 2062, 21827, 8865, 4092, 1966, 13908, 3135, 4667, 18692, 33351, 19442, 19371, 4878, 29998, 14868, 4740, 11154, 14557, 19203], 
  "k_local": [0, 1, 0, 2, 3, 0, 0, 4, 5, 6, 1, 2, 3, 4, 5, 6, 0, 7, 8], 
  "p_local": [9, 10, 11, 12, 13, 1, 7, 14, 15, 16, 17, 18, 2, 1, 0, 19, 20, 21, 22, 23], 
  "k_cluster": [0, 0, 1, 0, 0, 2, 3, 0, 0, 0, 2, 2, 2, 2, 2, 2, 4, 2, 2], 
  "p_cluster": [2, 2, 2, 2, 2, 3, 0, 2, 2, 2, 2, 2, 3, 1, 7, 2, 2, 2, 2, 2]
}
```

## Dataset Descriptions

* index: the index of the list.
* k_global: the global item ids of the known (input) list.
* p_global: the global item ids of the predicted (target) list.
* k_local: the local item ids of the known (input) list.
* p_local: the local item ids of the predicted (target) list.
* k_cluster: the cluster ids of the known (input) list.
* p_cluster: the cluster ids of the predicted (target) list.

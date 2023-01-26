# FANS: Fast Non-autoregressive Sequential Generation for Item List Continuation

TheWebConf 2023 accepted paper.

```bibtex
@inproceedings{liu2023fans, 
    title = {FANS: Fast Non-Autoregressive Sequence Generation for Item List Continuation},
    author = {Liu, Qijiong and Zhu, Jieming and Wu, Jiahao and Wu, Tiandeng and Dong, Zhenhua and Wu, Xiao-Ming},
    booktitle = {Proceedings of the ACM Web Conference 2023},
    month = {may},
    year = {2023},
    address = {Austin, Texas, USA}
}
```

## Datasets

Please refer to [data/readme.md](https://github.com/Jyonn/FANS/blob/master/data/README.md)

## Training

```shell
python worker.py 
  --config config/<DATASET>-bert-double-n10.yaml 
  --exp exp/curriculum-bert-double-step.yaml
```

## Testing

```shell
python worker.py 
  --config config/<DATASET>-bert-double-n10.yaml 
  --exp exp/test-curriculum-bert-double-step.yaml
```
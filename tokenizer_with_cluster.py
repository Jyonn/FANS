import argparse
import json
import os.path

from UniTok import UniTok, Column, Vocab
from UniTok.tok import BaseTok, IdTok
import pandas as pd
from tqdm import tqdm


class ListTok(BaseTok):
    def t(self, objs):
        ids = []
        for obj in objs:
            if self.pre_handler:
                obj = self.pre_handler(obj)
            ids.append(self.vocab.append(obj))
        return ids


class NumberVocab(Vocab):
    def append(self, obj) -> int:
        obj = str(obj)
        if obj not in self.obj2index:
            for o in range(self.get_size(), int(obj) + 1):
                self.obj2index[str(o)] = o
                self.index2obj[o] = str(o)
        return super(NumberVocab, self).append(obj)


class Tokenizer:
    def __init__(
            self,
            dataset: str,
            pred_portion=0.1,
            max_length=100,
            min_length=10,
            frequency=1,
            n_cluster=1000,
    ):
        self.dataset = dataset
        self.pred_portion = pred_portion
        self.max_length = max_length
        self.min_length = min_length
        self.frequency = frequency
        self.n_cluster = n_cluster
        self.store_dir = os.path.join('ListCont', f'{self.dataset.lower()}-n{self.n_cluster}')

    @staticmethod
    def get_vocabs():
        return Vocab(name='global_id'), NumberVocab(name='local_id'), Vocab(name='cluster_id')

    def get_cluster_vocabs(self):
        return [Vocab(name=f'cluster_{i}') for i in range(self.n_cluster)]

    def init_vocabs(self):
        self.global_vocab, self.local_vocab, self.cluster_vocab = self.get_vocabs()
        self.cluster_vocabs = self.get_cluster_vocabs()
        self.non_cluster = Vocab(name='non_cluster')

    def load_list_dict(self):
        print('LOAD list item data')
        list_dict = dict()
        with open(f'data/{self.dataset}.txt', 'r') as f:
            for line in f:
                if line.endswith('\n'):
                    line = line[:-1]
                if not line:
                    break

                list_id, item_id = line.split(' ')
                if list_id not in list_dict:
                    list_dict[list_id] = []
                list_dict[list_id].append(item_id)
        return list_dict

    def load_item_cluster_dict(self):
        return json.load(open(f'data/cluster/{self.dataset.lower()}-n{self.n_cluster}.json', 'r'))

    def list_item_formatter(self, list_dict, item_cluster_dict):
        index_series = []
        kg_series, pg_series = [], []  # known, pred global series
        kl_series, pl_series = [], []  # known, pred local series
        kc_series, pc_series = [], []  # known, pred cluster series

        index = 0
        for list_id in tqdm(list_dict):
            global_ = list_dict[list_id]
            cluster_ = list(map(item_cluster_dict.get, global_))
            local_ = []
            for i_, item in enumerate(global_):
                vocab = self.cluster_vocabs[cluster_[i_]] if cluster_[i_] is not None else self.non_cluster
                local_.append(vocab.append(item))

            for i in range(max(int(len(global_) // self.max_length), 1)):
                start = i * self.max_length
                stop = min((i + 1) * self.max_length, len(global_))

                c_global_ = global_[start:stop]
                c_local_ = local_[start:stop]
                c_cluster_ = cluster_[start:stop]
                if len(c_global_) < self.min_length:
                    break

                test_len = max(1, round(len(c_global_) * self.pred_portion))
                index_series.append(index)
                index += 1

                kg_series.append(c_global_[:-test_len])
                pg_series.append(c_global_[-test_len:])
                kl_series.append(c_local_[:-test_len])
                pl_series.append(c_local_[-test_len:])
                kc_series.append(c_cluster_[:-test_len])
                pc_series.append(c_cluster_[-test_len:])

        return pd.DataFrame(data=dict(
            index=index_series,
            k_global=kg_series,
            p_global=pg_series,
            k_local=kl_series,
            p_local=pl_series,
            k_cluster=kc_series,
            p_cluster=pc_series,
        ))

    def get_list_item_tok(self, df: pd.DataFrame, analyse=False):
        column_list = [
            ('global', self.global_vocab),
            ('local', self.local_vocab),
            ('cluster', self.cluster_vocab)
        ]

        tok = UniTok().add_col(Column(
            name='index',
            tokenizer=IdTok(name='index').as_sing()
        ))
        for column, vocab in column_list:
            tok.add_col(Column(
                name=f'k_{column}',
                tokenizer=ListTok(name=f'{column}s', vocab=vocab).as_list(
                    max_length=None if analyse else int(self.max_length * (1 - self.pred_portion)),
                    slice_post=True
                )
            )).add_col(Column(
                name=f'p_{column}',
                tokenizer=ListTok(name=f'{column}s', vocab=vocab).as_list(
                    max_length=None if analyse else max(int(self.max_length * self.pred_portion), 1),
                )
            ))

        return tok.read_file(df)

    def list_item_finetune(self, df, analyse, list_dict):
        unitok = self.get_list_item_tok(df, analyse)
        unitok.frequency_mode = False
        unitok.analyse()
        trim_before = self.global_vocab.get_size()
        self.global_vocab.trim_vocab(min_frequency=self.frequency)
        trim_after = self.global_vocab.get_size()
        print('***************** VOCAB', trim_before, trim_after)

        if trim_before == trim_after:
            return True, unitok, list_dict

        for list_id in tqdm(list_dict):
            list_ = list_dict[list_id]
            list_dict[list_id] = list(filter(lambda item_id: item_id in self.global_vocab.obj2index, list_))

        return False, unitok, list_dict

    def tokenize(self, analyse=False):
        print('START tokenize')
        list_dict = self.load_list_dict()
        item_cluster_dict = self.load_item_cluster_dict()
        index = 0
        while True:
            self.init_vocabs()
            index += 1
            df = self.list_item_formatter(list_dict, item_cluster_dict)
            print(f'***************** DF V{index}', df.size)
            finish, tok, list_dict = self.list_item_finetune(df, analyse, list_dict)
            print('FINISHED:', finish)
            if finish:
                break

        if analyse:
            return tok.analyse()
        tok.tokenize().store_data(os.path.join(self.store_dir))

        # saving cluster toks
        print('Non cluster size', self.non_cluster.get_size())
        cluster_store_dir = os.path.join(self.store_dir, 'clusters')
        os.makedirs(cluster_store_dir, exist_ok=True)
        cluster_vocabs = []
        for index in range(self.cluster_vocab.get_size()):
            c_index = int(self.cluster_vocab.index2obj[index])
            vocab = self.cluster_vocabs[c_index]
            vocab.name = f'cluster_{index}'
            vocab.save(cluster_store_dir)
            cluster_vocabs.append(vocab.get_size())
        print('total items', sum(cluster_vocabs))
        json.dump(cluster_vocabs, open(os.path.join(cluster_store_dir, 'cluster_vocab.json'), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--analyse', type=int, default=0)
    parser.add_argument('--portion', type=float)
    parser.add_argument('--frequency', type=int, default=1)
    parser.add_argument('--min_len', type=int, default=10)
    parser.add_argument('--n_cluster', type=int, default=1000)

    args = parser.parse_args()
    args.analyse = bool(args.analyse)
    print(args)

    Tokenizer(
        dataset=args.dataset,
        max_length=args.max_len,
        min_length=args.min_len,
        pred_portion=args.portion,
        frequency=args.frequency,
        n_cluster=args.n_cluster,
    ).tokenize(analyse=args.analyse)


# python tokenizer_with_cluster.py --dataset Zhihu --max_len 200 --min_len 10 --portion 0.5 --analyse 0 --frequency 10 --n_cluster 1000
# python tokenizer_with_cluster.py --dataset Goodreads --max_len 300 --min_len 20 --portion 0.5 --analyse 0 --frequency 10 --n_cluster 1000
# python tokenizer_with_cluster.py --dataset Spotify --max_len 300 --min_len 20 --portion 0.5 --analyse 0 --frequency 10 --n_cluster 1000
# python tokenizer_with_cluster.py --dataset AotM --max_len 100 --min_len 10 --portion 0.5 --analyse 0 --frequency 10 --n_cluster 889

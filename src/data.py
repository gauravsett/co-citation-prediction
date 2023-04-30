from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
from itertools import combinations
from torch_geometric.data import Dataset as GeoDataset
from torch_geometric.data import Data as GeoData
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class EncoderDataset(Dataset):

    def __init__(self, data_path, tokenizer_name):
        self.data = self._load_data(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, max_length=512
        )

    def _load_data(self, data_path):
        data = pd.read_feather(data_path)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]["text"]

    def collate_fn(self, batch):
        tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        return torch.tensor(tokens["input_ids"]), torch.tensor(tokens["attention_mask"])


class GraphDataset(GeoDataset):

    def __init__(self, embeddings, data):
        self.data = self._construct_graph(embeddings, data)

    def _construct_graph(self, embeddings, data):
        x = torch.tensor(embeddings)
        edge_index = torch.tensor(
            data[["id", "references"]].explode("references").values.t(),
            dtype=torch.long
        )
        data = GeoData(x=x, edge_index=edge_index, is_directed=True)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]


class RegressionDataset(torch.utils.data.Dataset):
    
    def __init__(self, data):
        data = data.set_index("id")
        co_cite_count = self._get_co_citations(data)
        print("co_cite_count", len(co_cite_count))
        co_cite_yearly = self._get_normalization(data, co_cite_count)
        print("co_cite_yearly", len(co_cite_yearly))
        self.examples = self._create_examples(
            data, co_cite_count, co_cite_yearly
        )
        print("examples", len(self.examples))

    def _get_co_citations(self, data) -> dict:
        co_cite_count = dict()
        for row in data.itertuples():
            count = 0
            for combination in combinations(row.references, 2):
                if combination[0] == combination[1]:
                    continue
                c = frozenset(combination)
                if c in co_cite_count:
                    co_cite_count[c] += 1
                else:
                    co_cite_count[c] = 1
                count += 1
                if count == 10:
                    break
        return co_cite_count

    def _get_normalization(self, data, co_cite_count):
        co_cite_yearly_sum = dict()
        co_cite_yearly_count = dict()
        for k, v in co_cite_count.items():
            k_1, k_2 = list(k)
            if k_1 not in data.index or k_2 not in data.index:
                continue
            year = data.loc[[k_1, k_2]]["year"].max()
            if year in co_cite_yearly_sum:
                co_cite_yearly_sum[year] += v
                co_cite_yearly_count[year] += 1
            else:
                co_cite_yearly_sum[year] = v
                co_cite_yearly_count[year] = 1
        co_cite_yearly = {
            k: v / co_cite_yearly_count[k]
            for k, v in co_cite_yearly_sum.items()
        }
        return co_cite_yearly
    
    def _create_examples(self, data, co_cite_count, co_cite_yearly):
        examples = []
        for k, v in co_cite_count.items():
            k_1, k_2 = list(k)
            if k_1 in data.index and k_2 in data.index:
                year = data.loc[[k_1, k_2]]["year"].max()
                v = v / co_cite_yearly[year]
            examples.append([(k_1, k_2), v])
            if k_1 in data.index:
                references = data.loc[k_1].references
                count = 0
                for k_3 in references:
                    if type(k_3) == str and frozenset((k_2, k_3)) not in co_cite_count:
                        examples.append([(k_1, k_3), 0])
                        count += 1
                        if count == 10:
                            break
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_fn_regression(batch, embeddings):
    x = torch.tensor(
        [(embeddings[k_1], embeddings[k_2],) for (k_1, k_2), _ in batch]
    )
    y = torch.tensor([y for _, y in batch])
    return x, y

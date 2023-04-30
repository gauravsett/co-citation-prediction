from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
from itertools import combinations
from torch_geometric.data import Dataset as GeoDataset
from torch_geometric.data import Data as GeoData
from torch.nn.utils.rnn import pad_sequence


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
    
    def __init__(self, data, embeddings):
        co_cite_count = self._get_co_citations(data)
        co_cite_yearly = self._get_normalization(data, co_cite_count)
        self.examples = self._create_examples(
            data, co_cite_count, co_cite_yearly, embeddings
        )

    def _get_co_citations(self, data) -> dict:
        co_cite_count = dict()
        for row in data.itertuples():
            for combination in combinations(row.references, 2):
                if combination[0] == combination[1]:
                    continue
                c = frozenset(combination)
                if c in co_cite_count:
                    co_cite_count[c] += 1
                else:
                    co_cite_count[c] = 1
        return co_cite_count

    def _get_normalization(self, data, co_cite_count):
        co_cite_yearly = dict()
        for k, v in co_cite_count.items():
            k_1, k_2 = list(k)
            year = data.loc[data["id"].isin([k_1, k_2])]["year"].max()
            if year in co_cite_yearly:
                co_cite_yearly[year].append(v)
            else:
                co_cite_yearly[year] = [v]
        co_cite_yearly = {
            k: (torch.tensor(v).mean(), torch.tensor(v).std())
            for k, v in co_cite_yearly.items()
        }
        return co_cite_yearly
    
    def _create_examples(self, data, co_cite_count, co_cite_yearly):
        examples = []
        for k, v in co_cite_count.items():
            k_1, k_2 = list(k)
            year = data.loc[data["id"].isin([k_1, k_2])]["year"].max()
            if year in co_cite_yearly:
                v = (v - co_cite_yearly[year][0]) / co_cite_yearly[year][1]
            examples.append([(k_1, k_2), v])
            for k_3 in data.loc[data["id"] == k_1].references:
                if frozenset([k_2, k_3]) not in co_cite_count:
                    examples.append([(k_1, k_3), 0])
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

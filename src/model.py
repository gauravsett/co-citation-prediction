# import torch
# import torch.nn as nn


class Model(nn.Module):
    
    def __init__(
            self, 
            encoder_model, 
            encoder_data,
            graph_model, 
            regression_model
        ):
        super(Model, self).__init__()
        self.encoder_model = encoder_model
        self.graph_model = graph_model
        self.regression_model = regression_model

    def initialize_embeddings(self, tokens):
        self.embeddings = self.encoder_model(tokens)

    def forward(self):
        if self.embeddings is None:
            raise ValueError("Embeddings not initialized")
        x = self.graph_model(self.embeddings)
        return self.regression_model(x)
    
    def train(self):
        pass

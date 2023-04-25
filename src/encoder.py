import torch
import torch.nn as nn
import transformers


class EncoderModel(nn.Module):
    
    def __init__(self, model_name, config):
        super(EncoderModel, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(
            model_name, config=config
        )
        self.config = config
        
    def forward(self, x):
        return self.model.forward(x)



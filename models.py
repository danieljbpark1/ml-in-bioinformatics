import torch
import torch.nn as nn
import torch.nn.Functional as F

class MLP(nn.Module):
    def __init__(self, hidden_layer_size: int):
        """Initializes a multi-layer perceptron model.
        
        Args:
            hidden_layer_size (int): Size of hidden layer.
        """
        super().__init__()
        
        # one batch of samples has dimensions [B, 101, 4]
        # where B is the number of samples in the batch
        self.flatten = nn.Flatten(start_dim=1)
        self.hidden_layer = nn.Linear(in_features=404, out_features=hidden_layer_size)
        self.output_layer = nn.Linear(in_features=hidden_layer_size + 1, out_features=1)
    
    def forward(self, x_batch: torch.Tensor, a_batch: torch.Tensor):
        """Performs forward pass with a batch of data.
        
        Args:
            x_batch (torch.Tensor): Batch of Chr22 segments. Tensor size [B, 101, 4]
            a_batch (torch.Tensor): Batch of accessibility values per segment. Tensor size [B, 1]
        
        """
        x = self.flatten(x_batch)  # x now has size [B, 404]
        
        x = self.hidden_layer(x)  # [B, hidden_layer_size]
        x = F.relu(x)
        x = F.dropout(input=x, p=0.3)
        
        x = torch.cat((x, a_batch), dim=1)  # [B, hidden_layer_size + 1]
        
        x = self.output_layer(x)  # [B, 1]
        
        return x

class LSTM(nn.Module):
    def __init__(self, lstm_hidden_layer_size: int, mlp_hidden_layer_size: int):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=4, 
            hidden_size=lstm_hidden_layer_size,
            batch_first=True,
        )
        
        self.mlp_1 = nn.Linear(
            in_features=lstm_hidden_layer_size,
            out_features=mlp_hidden_layer_size,
        )
        
        self.mlp_2 = nn.Linear(
            in_features=mlp_hidden_layer_size + 1,
            out_features=1,
        )
        
       
    def forward(self, batch_x: torch.Tensor, batch_a: torch.Tensor):
        """Performs forward pass with a batch of data.
        
        Args:
            batch_x (torch.Tensor): Batch of Chr22 segments. Tensor size [B, 101, 4]
            batch_a (torch.Tensor): Batch of accessibility values per segment. Tensor size [B, 1]
        """
        
        # output size [B, 101, lstm_hidden_layer_size]
        # hn size [1, B, lstm_hidden_layer_size]
        # cn size [1, B, lstm_hidden_layer_size]
        output, (hn, cn) = self.lstm(batch_x)  
        
        lstm_last_hidden_layer = torch.squeeze(input=hn, dim=0)  # [B, lstm_hidden_layer_size]
        
        x = self.mlp_1(lstm_last_hidden_layer)  # [B, mlp_hidden_layer_size]
        x = F.relu(x)
        x = F.dropout(input=x, p=0.3)
        
        x = torch.cat((x, batch_a), dim=1)  # [B, mlp_hidden_layer_size + 1]
        
        x = self.mlp_2(x)  # [B, 1]
        
        return x
        

class CNN(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_layer_size: int):
        """Initializes a multi-layer perceptron model.
        
        Args:
            hidden_layer_size (int): Size of hidden layer.
        """
        super().__init__()
        
        # one batch of samples has dimensions [B, 101, 4]
        # where B is the number of samples in the batch
        self.hidden_layer = nn.Linear(in_features=404, out_features=hidden_layer_size)
        self.output_layer = nn.Linear(in_features=hidden_layer_size + 1, out_features=1)
    
    def forward(self, x_batch: torch.Tensor, a_batch: torch.Tensor):
        """Performs forward pass with a batch of data.
        
        Args:
            x_batch (torch.Tensor): Batch of Chr22 segments. Tensor size [B, 101, 4]
            a_batch (torch.Tensor): Batch of accessibility values per segment. Tensor size [B, 1]
        
        """
        x = torch.flatten(x_batch, start_dim=1)  # x now has size [B, 404]
        
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
    def __init__(
        self, 
        conv_layer_1_num_channels,
        conv_layer_1_kernel_size,
        max_pool_layer_1_kernel_size,
        conv_layer_2_num_channels,
        conv_layer_2_kernel_size,
        max_pool_layer_2_kernel_size,
        mlp_hidden_layer_size,
    ):
        self.conv_layer_1 = nn.Conv1d(
            in_channels=4,
            out_channels=conv_layer_1_num_channels,
            kernel_size=conv_layer_1_kernel_size,
        )

        self.conv_layer_2 = nn.Conv1d(
            in_channels=conv_layer_1_num_channels,
            out_channels=conv_layer_2_num_channels,
            kernel_size=conv_layer_2_kernel_size,
        )

        self.max_pool_layer_1_kernel_size = max_pool_layer_1_kernel_size
        self.max_pool_layer_2_kernel_size = max_pool_layer_2_kernel_size

        conv_layer_1_length = 101 - (conv_layer_1_kernel_size - 1)
        max_pool_layer_1_length = math.floor((conv_layer_1_length - (max_pool_layer_1_kernel_size - 1) - 1) / max_pool_layer_1_kernel_size + 1)
        conv_layer_2_length = max_pool_layer_1_length - (conv_layer_2_kernel_size - 1)
        max_pool_layer_2_length = math.floor((conv_layer_2_length - (max_pool_layer_2_kernel_size - 1) - 1) / max_pool_layer_2_kernel_size + 1)

        self.mlp_1 = nn.Linear(
            in_features=conv_layer_2_num_channels * max_pool_layer_2_length,
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
        x = torch.swapaxes(input=batch_x, axis0=1, axis1=2)  # [B, 4, 101]
        
        x = self.conv_layer_1(x) 
        x = F.relu(x)
        x = F.dropout1d(x, p=0.25)
        x = F.max_pool1d(x, kernel_size=self.max_pool_layer_1_kernel_size)

        x = self.conv_layer_2(x) 
        x = F.relu(x)
        x = F.dropout1d(x, p=0.25)
        x = F.max_pool1d(x, kernel_size=self.max_pool_layer_2_kernel_size)

        x = torch.flatten(x, start_dim=1)

        x = self.mlp_1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)

        x = torch.cat((x, batch_a), dim=1)

        x = self.mlp_2(x)

        return x

"""
Main model file
"""

import torch
import torch.nn as nn

#--------------------------------------------------------------------------------------------------
# Various models

class Base_Model(nn.Module):
    """
    Simple conv transformer model

    """
    def __init__(self, num_input_features=1, d_model=64, kernel_size=15, dna=False):
        super().__init__()

        self.dna = dna
        hidden_dim = d_model+d_model//2 if dna else d_model

        # 1. Input is (B, num_input_features, L, n_fibers)
        # We use a kernel of (K, 1) to process each fiber independently
        self.signal_encoder = nn.Sequential(
            nn.Conv2d(num_input_features, d_model, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0)),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        self.dna_encoder = nn.Sequential(
            nn.Conv1d(4, d_model//2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model//2),
            nn.GELU(),
        )

        self.concat_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size, kernel_size), padding=(kernel_size//2, 0)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Reduces (B, d_model*(3/2), L, n_fibers) -> (B, d_model, 1, 1)
            nn.Flatten(),           # -> (B, d_model*(3/2))
            nn.Linear(hidden_dim, d_model),  # -> (B, d_model)
            nn.GELU(),
            nn.Linear(d_model, 1),  # -> (B, 1)
            nn.Sigmoid(),           # -> (B, 1) where values are 0.0 to 1.0 (percentages)
        )

    def forward(self, fibers, dna):
    # fibers: (B, C, L, N), dna: (B, L, 4)
        if self.dna:
            x1 = self.signal_encoder(fibers)
            dna_e = self.dna_encoder(dna)
            # dna_e = dna_e.unsqueeze(-1).repeat(1, 1, 1, x1.shape[-1])
            dna_e = dna_e.unsqueeze(-1)  # Shape: [B, C, L, 1]
            dna_e = dna_e.expand(-1, -1, -1, x1.shape[-1]) # Shape: [B, C, L, N]
            x1 = torch.cat((x1,dna_e), dim=1)
            x2 = self.concat_encoder(x1)
            x3 = self.regressor(x2)
        else:
            x1 = self.signal_encoder(fibers)
            x2 = self.concat_encoder(x1)
            x3 = self.regressor(x2)

        return x3 * 100

#--------------------------------------------------------------------------------------------------
# model selection based on cmd arg

def model_selector(model_arg, args):

    model_name = model_arg.lower()

    if model_name=="base": return Base_Model(num_input_features=args.num_input_features, d_model=args.d_model, dna=args.cat_dna)

    raise NotImplementedError(f"Model not implemented: {model_arg}")


#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()

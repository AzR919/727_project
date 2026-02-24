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
    def __init__(self, num_input_features=1, d_model=64, kernel_size=15):
        super().__init__()

        # 1. Input is (B, num_input_features, L, n_fibers)
        # We use a kernel of (K, 1) to process each fiber independently
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_features, d_model, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0)),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0)),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Reduces (B, d_model, L, n_fibers) -> (B, d_model, 1, 1)
            nn.Flatten(),           # -> (B, d_model)
            nn.Linear(d_model, 1),  # -> (B, 1)
            nn.Sigmoid()            # -> (B, 1) where values are 0.0 to 1.0 (percentages)
        )

    def forward(self, fibers, dna):

        x1 = self.encoder(fibers)
        x2 = self.regressor(x1)

        return x2 * 100

#--------------------------------------------------------------------------------------------------
# model selection based on cmd arg

def model_selector(model_arg, args):

    model_name = model_arg.lower()

    if model_name=="base": return Base_Model(args.fibers_per_entry)

    raise NotImplementedError(f"Model not implemented: {model_arg}")


#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()

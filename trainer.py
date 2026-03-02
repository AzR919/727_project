"""
Main train loop

"""

import wandb
import torch
import torch.nn as nn

from utils import *
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, dataset, epochs=10,
                 batch_size=32, lr=1e-4, patience=5,
                 run_name="debug", config={}):

        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.iters_per_epoch = dataset.iters_per_epoch
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=patience
        )
        self.criterion = nn.MSELoss()

        # Start a new wandb run to track this script.
        self.wandb_run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="liblab",
            # Set the wandb project where this run will be logged.
            project="fiber_mixture",
            name=run_name,
            # Track hyperparameters and run metadata.
            config=config,
        )

        self.wandb_run.watch(self.model)
        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("target", step_metric="epoch")
        wandb.define_metric("predicted", step_metric="epoch")

        self.config = config

    def train_step(self, batch):

        self.model.train()
        fiber_tensor, dna, true_percentages = [b.to(self.device) for b in batch[:3]]
        self.optimizer.zero_grad()
        output = self.model(fiber_tensor, dna)
        loss = self.criterion(output, true_percentages[:,:1])
        loss.backward()
        self.optimizer.step()
        return loss.item(), output

    def train(self, save_dir):

        loader = DataLoader(self.dataset, batch_size=self.batch_size)

        losses = []
        rep_tar = []
        rep_out = []
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                loss, output = self.train_step(batch)
                total_loss += loss
            avg_loss = total_loss / self.iters_per_epoch
            self.scheduler.step(avg_loss)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            losses.append(avg_loss)
            if not (epoch % 10):
                tar = batch[2][0,0].item()
                out = output[0,0].item()
                rep_tar.append(tar)
                rep_out.append(out)
                self.wandb_run.log({"train_loss": avg_loss, "Target_instance": tar, "Output_instance": out, "epoch": epoch})
            else:
                self.wandb_run.log({"train_loss": avg_loss, "epoch": epoch})


        plot_reps(save_dir, self.wandb_run, rep_tar, rep_out, epoch+1)
        plot_loss(save_dir, losses, epoch+1)

#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()

"""
Main train loop

"""

import wandb
import torch
import torch.nn as nn

from utils import *
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, dataset, val_dataset=None, test_dataset=None,
                 epochs=10, batch_size=32, lr=1e-4, patience=5,
                 run_name="debug", config={}):

        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
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

        train_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size) if self.val_dataset else None

        losses = []
        rep_tar = []
        rep_out = []
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                loss, output = self.train_step(batch)
                total_loss += loss
            avg_train_loss = total_loss / self.iters_per_epoch
            losses.append(avg_train_loss)

            if not (epoch % 10):
                tar = batch[2][0,0].item()
                out = output[0,0].item()
                rep_tar.append(tar)
                rep_out.append(out)
                if self.val_dataset:
                    avg_val_loss = self.validate(val_loader, len(self.val_dataset))
                    self.scheduler.step(avg_val_loss) # Step scheduler on Val Loss
                    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                    self.wandb_run.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "Target_instance": tar, "Output_instance": out, "epoch": epoch})
                else:
                    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f}")
                    self.wandb_run.log({"train_loss": avg_train_loss, "Target_instance": tar, "Output_instance": out, "epoch": epoch})
            else:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}")
                self.wandb_run.log({"train_loss": avg_train_loss, "epoch": epoch})

        # Final Testing Phase
        if self.test_dataset:
            print("Running final test...")
            test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
            test_loss = self.validate(test_loader, len(self.test_dataset))
            print(f"Final Test Loss: {test_loss:.6f}")
            self.wandb_run.log({"test_loss": test_loss})

        plot_reps(save_dir, self.wandb_run, rep_tar, rep_out, epoch+1)
        plot_loss(save_dir, losses, epoch+1)

    # def train(self, save_dir):
    #     train_loader = DataLoader(self.dataset, batch_size=self.batch_size)
    #     val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size) if self.val_dataset else None

    #     losses = []
    #     for epoch in range(self.epochs):
    #         # Training Phase
    #         total_train_loss = 0
    #         for batch in train_loader:
    #             loss, output = self.train_step(batch)
    #             total_train_loss += loss

    #         avg_train_loss = total_train_loss / self.iters_per_epoch

    #         # Validation Phase (Every 10 epochs)
    #         avg_val_loss = None
    #         if val_loader and (epoch % 10 == 0 or epoch == self.epochs - 1):
    #             avg_val_loss = self.validate(val_loader)
    #             self.scheduler.step(avg_val_loss) # Step scheduler on Val Loss
    #             print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    #             self.wandb_run.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})
    #         else:
    #             print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f}")
    #             self.wandb_run.log({"train_loss": avg_train_loss, "epoch": epoch})

    #     # Final Testing Phase
    #     if self.test_dataset:
    #         print("Running final test...")
    #         test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
    #         test_loss = self.validate(test_loader)
    #         print(f"Final Test Loss: {test_loss:.6f}")
    #         self.wandb_run.log({"test_loss": test_loss})

    def validate(self, loader, total_iters):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loader:
                fiber_tensor, dna, true_percentages = [b.to(self.device) for b in batch[:3]]
                output = self.model(fiber_tensor, dna)
                loss = self.criterion(output, true_percentages[:,:1])
                val_loss += loss.item()
        return val_loss / total_iters if total_iters > 0 else 0

#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()

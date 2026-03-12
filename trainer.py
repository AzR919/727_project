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
                 epochs=10, batch_size=32, lr=1e-4, patience=9,
                 run_name="debug", config={}, save_dir="./results"):

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
            self.optimizer, mode='min', patience=patience,
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

        self.save_dir = save_dir
        self.best_val_loss = float('inf')
        self.model_save_path = os.path.join(save_dir, "best_model.pt")

        self.debug = config.debug
        if self.debug:
            self.epoch_history = []  # List of (val_loss, epoch_index)
            self.debug_dir = os.path.join(save_dir, "debug_dumps")
            os.makedirs(self.debug_dir, exist_ok=True)

    def train_step(self, batch):

        self.model.train()
        fiber_tensor, dna, true_percentages = [b.to(self.device) for b in batch[:3]]
        self.optimizer.zero_grad()
        output = self.model(fiber_tensor, dna)
        loss = self.criterion(output, true_percentages[:,:1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item(), output, true_percentages[:, :1].detach().cpu()

    def train(self):

        train_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size) if self.val_dataset else None

        losses = []
        rep_tar = []
        rep_out = []
        for epoch in range(self.epochs):
            total_loss = 0
            epoch_train_percs = []
            batches_processed = 0
            for batch in train_loader:
                loss, output, batch_percs = self.train_step(batch)
                total_loss += loss
                epoch_train_percs.append(batch_percs)
                batches_processed += 1

            # --- SAFETY CHECK ---
            if batches_processed == 0:
                print(f"Warning: Epoch {epoch} produced no batches. Skipping.")
                continue

            avg_train_loss = total_loss / batches_processed if batches_processed > 0 else 0
            train_percs_flat = torch.cat(epoch_train_percs, dim=0).numpy()
            losses.append(avg_train_loss)

            self.scheduler.step(avg_train_loss)

            log_dict = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_perc_dist": wandb.Histogram(train_percs_flat),
            }
            print_str = f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}"

            # logging
            if not (epoch % 5):
                tar = batch[2][0,0].item()
                out = output[0,0].item()
                rep_tar.append(tar)
                rep_out.append(out)
                log_dict["Target_instance"] = tar
                log_dict["Output_instance"] = out

                if self.val_dataset:
                    avg_val_loss, val_percs = self.validate(val_loader)
                    # self.scheduler.step(avg_val_loss) # Step scheduler on Val Loss

                    log_dict["val_loss"] = avg_val_loss
                    log_dict["val_perc_dist"] = wandb.Histogram(val_percs.numpy())

                    print_str = f"{print_str} | Val Loss: {avg_val_loss:.6f}"
                    # Check for improvement
                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': avg_val_loss,
                        }, self.model_save_path)
                        print(f"--> New Best Model Saved (Val Loss: {avg_val_loss:.6f})")

                        # Optional: Log the best loss to wandb as a summary metric
                        self.wandb_run.summary["best_val_loss"] = self.best_val_loss

                # Inside Trainer.train() loop after validation:
                if self.debug:
                    self.epoch_history.append((avg_val_loss, epoch))

                    # Sort history by loss: Worst (highest loss) first
                    self.epoch_history.sort(key=lambda x: x[0], reverse=True)

                    # If this epoch is currently one of the 3 worst:
                    worst_3_epochs = [e[1] for e in self.epoch_history[:3]]

                    if epoch in worst_3_epochs:
                        # 1. Save the Model Weights for this bad epoch
                        bad_model_path = os.path.join(self.debug_dir, f"model_epoch_{epoch}_bad.pt")
                        torch.save(self.model.state_dict(), bad_model_path)

                        # 2. Save the 10 worst batches
                        batch_dump_path = os.path.join(self.debug_dir, f"worst_batches_epoch_{epoch}.pt")
                        torch.save(self.current_epoch_worst, batch_dump_path)

                        print(f"!!! High Loss Detected. Saved Epoch {epoch} debug data.")

                        # 3. Cleanup: If we now have more than 3 "bad" files, delete the one that's no longer in top 3
                        # (Optional: Implementation depends on how much disk space you have)

            print(print_str)
            self.wandb_run.log(log_dict)

        final_model_path = os.path.join(self.save_dir, "final_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_train_loss,
        }, final_model_path)

        # Final Testing Phase
        if self.test_dataset:
            # 1. Run Test on Final Model (the weights currently in memory)
            print("Running test on FINAL model weights...")
            test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
            final_test_loss, final_test_percs = self.validate(test_loader)
            self.wandb_run.log({
                "test_loss_final": final_test_loss,
                "test_perc_dist_final": wandb.Histogram(final_test_percs.numpy())
            })

            # 2. Run Test on Best Model (the weights saved to disk)
            if os.path.exists(self.model_save_path):
                print(f"Loading best model from {self.model_save_path} for testing...")
                checkpoint = torch.load(self.model_save_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])

                print("Running test on BEST model weights...")
                best_test_loss, best_test_percs = self.validate(test_loader)
                self.wandb_run.log({
                    "test_loss_best": best_test_loss,
                    "test_perc_dist_best": wandb.Histogram(best_test_percs.numpy())
                })
                print(f"Final Weights Loss: {final_test_loss:.4f} vs Best Weights Loss: {best_test_loss:.4f}")
            else:
                print(f"Best model file not found, skipping best-model evaluation. Final Weights Loss: {final_test_loss:.4f}")

        plot_reps(self.save_dir, self.wandb_run, rep_tar, rep_out, epoch+1)
        plot_loss(self.save_dir, losses, epoch+1)

    def validate(self, loader):
        self.model.eval()
        val_loss = 0
        all_true_percs = []
        batches_processed = 0

        # Buffer to store batches for the current epoch if debugging
        epoch_batches = []

        with torch.no_grad():
            for batch in loader:
                fiber_tensor, dna, true_percs, loci = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]

                output = self.model(fiber_tensor, dna)
                # Calculate individual losses for each sample in the batch to be precise
                # We'll use the mean loss of the batch for sorting
                batch_loss = self.criterion(output, true_percs[:,:1]).item()

                if self.debug:
                    # Store batch data (moving to CPU to save VRAM)
                    epoch_batches.append({
                        'loss': batch_loss,
                        'fiber_tensor': fiber_tensor.cpu(),
                        'dna': dna.cpu(),
                        'targets': true_percs[:,:1].cpu(),
                        'outputs': output.cpu(),
                        'locus': loci
                    })

                val_loss += batch_loss
                all_true_percs.append(true_percs[:, :1].cpu())
                batches_processed += 1

        avg_loss = val_loss / batches_processed if batches_processed > 0 else 0
        combined_percs = torch.cat(all_true_percs, dim=0) if all_true_percs else torch.tensor([])

        if self.debug:
            # Sort by loss descending and keep top 10
            epoch_batches.sort(key=lambda x: x['loss'], reverse=True)
            self.current_epoch_worst = epoch_batches[:10]

        return avg_loss, combined_percs

#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()

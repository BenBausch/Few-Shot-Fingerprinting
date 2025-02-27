import copy
import logging
import os

import torch
from monai.networks.nets import ViT
from tqdm import tqdm

import wandb
from plotting.triplets import plot_triplets
from plotting.tsne import plot_tsne
from utils.utils import dict_to_device


class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        validation_loader,
        num_epochs,
        device,
        result_path,
        checkpoint,
        validation_tester,
        regularization_loss=None,
        log_frequency=20,
        sweep=False,
    ):
        self.config = config
        self.plotting_config = config["PLOTTING_TRAINING"]
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_epochs = num_epochs
        self.device = device
        self.result_path = result_path
        self.result_train_path = os.path.join(result_path, "train")
        self.checkpoint = checkpoint
        self.current_epoch = 0
        self.validation_result_path = validation_tester.result_path
        self.validation_tester = validation_tester
        self.regularization_loss = regularization_loss
        self.log_frequency = log_frequency
        self.sweep = sweep

        # torch automatic mixed precision
        self.scaler = torch.GradScaler()

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.result_train_path, exist_ok=True)
        self.load_checkpoint()
        wandb.watch(self.model, log_freq=1, log="gradients")

    def save_model_checkpoint(self):
        """
        Saves the training checkpoint
        """
        # Save checkpoint with model, optimizer, and other info
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(
            self.result_train_path, f"model_checkpoint_{self.current_epoch}.pth"
        )

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self):
        """
        Load the training checkpoint if provided
        """
        if self.checkpoint is not None:
            self.current_epoch = self.checkpoint["epoch"]

    def fit(self):
        self.best_epoch = 0
        self.best_metric = 0.0
        self.best_model = None
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train()
            with torch.no_grad():
                new_val_path = os.path.join(
                    self.validation_result_path, f"model_checkpoint_{epoch}"
                )
                os.makedirs(new_val_path, exist_ok=True)
                self.validation_tester.result_path = new_val_path
                self.validation_tester.model = self.model
                current_val = self.validate()
            if current_val > self.best_metric:
                self.best_metric = current_val
                self.best_epoch = epoch
                self.best_model = copy.deepcopy(self.model)
            self.save_model_checkpoint()
        return self.best_model, self.best_epoch

    def train(self):
        self.model.train()
        self.running_loss = 0.0
        self.running_pos = 0.0
        self.running_neg = 0.0
        self.avg_epoch_loss = 0.0
        for idx, (data, meta_data) in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):
            data = dict_to_device(data, self.device)
            anchor, positive = data["anchor_img"], data["positive_img_0"]

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                # This unpacks the embeddings when the output of the model is a tuple (e.g. ViT)
                anchor_embedding = self.model(anchor)
                positive_embedding = self.model(positive)

                if isinstance(self.model, ViT):
                    anchor_embedding = anchor_embedding[0]
                    positive_embedding = positive_embedding[0]

                # return indices for plotting
                loss, pos, neg, triplet_indexes = self.criterion(
                    anchor_embedding,
                    positive_embedding,
                    meta_data["anchor_id"],
                    True,
                )

                if torch.isnan(loss):
                    raise ValueError("Loss is NaN")

            if self.regularization_loss is not None:
                loss += self.regularization_loss(self.model.parameters())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            self.running_loss += loss.item()
            self.running_pos += pos.item()
            self.running_neg += neg.item()
            self.avg_epoch_loss += loss.item()
            self.log_train_metrics(idx)
        self.optimizer.zero_grad(set_to_none=True)
        self.avg_epoch_loss /= len(self.train_loader)
        logging.info(
            f"Epoch {self.current_epoch} - Average Loss: {self.avg_epoch_loss}"
        )
        wandb.log(
            {"train_loss": self.avg_epoch_loss, "epoch": self.current_epoch},
            step=self.current_epoch,
        )
        if not self.sweep and self.plotting_config["2D_TSNE_WITH_IMAGES"]["PLOT"]:
            # plot 2d tsne with images
            plot_tsne(
                result_path=self.result_train_path,
                anchor_embeddings=anchor_embedding.detach().cpu().numpy(),
                positive_embeddings=positive_embedding.detach().cpu().numpy(),
                anchor_imgs=data["anchor_img"].cpu().numpy(),
                positive_imgs=data["positive_img_0"].cpu().numpy(),
                epoch=self.current_epoch,
                dims=2,
            )
        if not self.sweep and self.plotting_config["3D_TSNE"]["PLOT"]:
            # plot 3d tsne
            plot_tsne(
                result_path=self.result_train_path,
                anchor_embeddings=anchor_embedding.detach().cpu().numpy(),
                positive_embeddings=positive_embedding.detach().cpu().numpy(),
                epoch=self.current_epoch,
                dims=3,
            )
        if not self.sweep and self.plotting_config["TRIPLETS"]["PLOT"]:
            plot_triplets(
                result_path=self.result_train_path,
                anchor_imgs=data["anchor_img"].cpu().numpy(),
                positive_imgs=data["positive_img_0"].cpu().numpy(),
                epoch=self.current_epoch,
                triplet_indices=triplet_indexes,
            )

    def validate(self):
        self.model.eval()
        (avg_recall_at_k_shot, avg_topk_hit, avg_majority_vote_hit_at_k_shot) = (
            self.validation_tester.test()
        )
        # Log the metrics
        wandb.log(
            {
                "avg_recall_at_k_shot": avg_recall_at_k_shot,
                "avg_topk_hit": avg_topk_hit,
                "avg_majority_vote_hit_at_k_shot": avg_majority_vote_hit_at_k_shot,
                "epoch": self.current_epoch,
            },
            step=self.current_epoch,
        )

        return avg_topk_hit[1]

    def log_train_metrics(self, idx):
        if (idx + 1) % self.log_frequency == 0:
            wandb.log(
                {
                    "train_loss": self.running_loss / self.log_frequency,
                    "positive_loss/distance": self.running_pos / self.log_frequency,
                    "negative_loss/distance": self.running_neg / self.log_frequency,
                    "epoch": self.current_epoch,
                },
                step=self.current_epoch,
            )
            self.running_loss = 0.0
            self.running_pos = 0.0
            self.running_neg = 0.0

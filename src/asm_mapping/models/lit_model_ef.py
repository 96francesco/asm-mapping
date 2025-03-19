# mypy: allow-untyped-defs
import torchmetrics
import segmentation_models_pytorch as smp # type: ignore
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection

class LitModelEarlyFusion(pl.LightningModule):
      """
      PyTorch Lightning module for early fusion of PlanetScope and Sentinel-1 data.
      
      This model implements input-level fusion of PlanetScope optical imagery and 
      Sentinel-1 SAR data. It concatenates both data sources along the channel dimension
      before passing them through a single encoder-decoder network.
      """
      def __init__(self,
                 s1_in_channels: int = 3,
                 planet_in_channels: int = 7,
                 lr: float = 1e-3, 
                 threshold: float = 0.5,
                 weight_decay: float = 1e-5,
                 alpha: float = 0.25,
                 gamma: float = 2.0):
            """
            Initialize the Early Fusion model.
            
            Args:
                  s1_in_channels (int): Number of input channels for Sentinel-1 data
                  planet_in_channels (int): Number of input channels for PlanetScope data
                  lr (float): Learning rate for optimizer
                  threshold (float): Threshold for binary classification
                  weight_decay (float): Weight decay for optimizer
                  alpha (float): Alpha parameter for Focal Loss
                  gamma (float): Gamma parameter for Focal Loss
            """
            super().__init__()
            self.s1_in_channels = s1_in_channels
            self.planet_in_channels = planet_in_channels
            self.combined_in_channels = s1_in_channels + planet_in_channels
            self.lr = lr
            self.threshold = threshold
            self.weight_decay = weight_decay
            self.alpha = alpha
            self.gamma = gamma
            self.optimizer = torch.optim.Adam
            self.criterion = smp.losses.FocalLoss(alpha=alpha, gamma=gamma, mode='binary')
            
            # initialize the U-Net model with combined input channels
            self.model = smp.Unet(
                  encoder_name="resnet18",
                  decoder_attention_type="scse",
                  decoder_use_batchnorm=True,
                  in_channels=self.combined_in_channels,
                  classes=1
            )
            
            # initialize metrics collection
            metrics = MetricCollection({
                  'accuracy': torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2, threshold=self.threshold),
                  'precision_macro': torchmetrics.Precision(task='multiclass', average='macro', num_classes=2, threshold=self.threshold),
                  'precision_asm': torchmetrics.Precision(task='multiclass', average='none', num_classes=2, threshold=self.threshold),
                  'recall_macro': torchmetrics.Recall(task='multiclass', average='macro', num_classes=2, threshold=self.threshold),
                  'recall_asm': torchmetrics.Recall(task='multiclass', average='none', num_classes=2, threshold=self.threshold),
                  'f1_score_macro': torchmetrics.F1Score(task='multiclass', average='macro', num_classes=2, threshold=self.threshold),
                  'f1_score_asm': torchmetrics.F1Score(task='multiclass', average='none', num_classes=2, threshold=self.threshold)
            })
            
            self.train_metrics = metrics.clone(prefix='train_')
            self.val_metrics = metrics.clone(prefix='val_')
            self.test_metrics = metrics.clone(prefix='test_')
            
            self.save_hyperparameters()

      def forward(self, planet_input: torch.Tensor, s1_input: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the model.
            
            Args:
                  planet_input (torch.Tensor): PlanetScope input tensor
                  s1_input (torch.Tensor): Sentinel-1 input tensor
                  
            Returns:
                  torch.Tensor: Model output
            """
            # concatenate inputs along the channel dimension
            combined_input = torch.cat([planet_input, s1_input], dim=1)
            return self.model(combined_input)

      def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                        batch_idx: int) -> torch.Tensor:
            """
            Training step for the model.
            
            Args:
                  train_batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 
                        Tuple containing (planet_input, s1_input, target)
                  batch_idx (int): Batch index
                  
            Returns:
                  torch.Tensor: Loss value
            """
            planet_input, s1_input, y = train_batch
            outputs = self(planet_input, s1_input)
            y = y.unsqueeze(1).type_as(planet_input)
            loss = self.criterion(outputs, y)
            
            # Calculate and log metrics
            probs = torch.sigmoid(outputs)
            preds = (probs > self.threshold).float()
            self.train_metrics.update(preds, y)
            self.log('train_loss', loss, prog_bar=True, on_step=False, 
                  on_epoch=True, sync_dist=True)
            
            return loss

      def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                        batch_idx: int) -> torch.Tensor:
            """
            Validation step for the model.
            
            Args:
                  val_batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 
                        Tuple containing (planet_input, s1_input, target)
                  batch_idx (int): Batch index
                  
            Returns:
                  torch.Tensor: Loss value
            """
            planet_input, s1_input, y = val_batch
            y = y.unsqueeze(1).type_as(planet_input)
            outputs = self(planet_input, s1_input)
            loss = self.criterion(outputs, y)
            
            # Calculate and log metrics
            probs = torch.sigmoid(outputs)
            preds = (probs > self.threshold).float()
            self.val_metrics.update(preds, y)
            
            metrics = self.val_metrics.compute()
            log_dict = {'val_loss': loss}
            for k, v in metrics.items():
                  if isinstance(v, torch.Tensor) and v.numel() > 1:
                        log_dict[k] = v[1]
                  else:
                        log_dict[k] = v
            
            self.log_dict(log_dict, prog_bar=True, on_step=False, 
                        on_epoch=True, sync_dist=True)
            self.val_metrics.reset()
            
            return loss

      def test_step(self, test_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                        batch_idx: int) -> torch.Tensor:
            """
            Test step for the model.
            
            Args:
                  test_batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 
                        Tuple containing (planet_input, s1_input, target)
                  batch_idx (int): Batch index
                  
            Returns:
                  torch.Tensor: Loss value
            """
            planet_input, s1_input, y = test_batch
            y = y.unsqueeze(1).type_as(planet_input)
            outputs = self(planet_input, s1_input)
            loss = self.criterion(outputs, y)
            
            # Calculate and log metrics
            probs = torch.sigmoid(outputs)
            preds = (probs > self.threshold).float()
            self.test_metrics.update(preds, y)
            
            metrics = self.test_metrics.compute()
            log_dict = {'test_loss': loss}
            for k, v in metrics.items():
                  if isinstance(v, torch.Tensor) and v.numel() > 1:
                        log_dict[k] = v[1]
                  else:
                        log_dict[k] = v
                  
            self.log_dict(log_dict, sync_dist=True)
            self.test_metrics.reset()
            
            return loss

      def configure_optimizers(self):
            """
            Configure the optimizer and learning rate scheduler.
            
            Returns:
                  dict: Optimizer configuration
            """
            optimizer = self.optimizer(self.parameters(), 
                                    lr=self.lr, 
                                    weight_decay=self.weight_decay)

            scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                          factor=0.5, patience=8)
            return {
                  "optimizer": optimizer,
                  "lr_scheduler": {
                  "scheduler": scheduler,
                  "monitor": "val_loss",
                  "frequency": 1
                  }
            }
# mypy: allow-untyped-defs
import torchmetrics
import segmentation_models_pytorch as smp # type: ignore
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection

class LitModelStandalone(pl.LightningModule):
      """
      # TODO: fix docstring
      """
      def __init__(self, model: Optional[nn.Module] = None, weight_decay: float =1e-5,
                 lr: float = 1e-3, threshold: float = 0.5, in_channels: int = 6,
                 alpha: float  =0.25, gamma: float = 2.0):
            super().__init__()
            self.weight_decay: float = weight_decay
            self.lr: float = lr
            self.threshold: float = threshold
            self.alpha: float = alpha
            self.gamma: float = gamma
            self.optimizer = torch.optim.Adam
            self.criterion: nn.Module = smp.losses.FocalLoss(alpha=alpha, gamma=gamma, mode='binary')
            self.save_hyperparameters()

            if model is None:
                  # initialize 'standard' unet
                  # None is used when loading model from checkpoints
                  self.model = smp.Unet(
                        encoder_name="resnet34",
                        decoder_use_batchnorm=True,
                        decoder_attention_type='scse',
                        encoder_weights=None,
                        in_channels=in_channels,
                        classes=1
                  )
                  pass
            else:
                  self.model = model
     
            # initiale metric collection
            metrics = MetricCollection({
                  'accuracy': torchmetrics.Accuracy(task='binary', average='macro', threshold=self.threshold),
                  'precision_macro': torchmetrics.Precision(task='binary', average='macro', threshold=self.threshold),
                  'precision_asm': torchmetrics.Precision(task='binary', average=None, threshold=self.threshold, num_classes=2),
                  'recall_macro': torchmetrics.Recall(task='binary', average='macro', threshold=self.threshold),
                  'recall_asm': torchmetrics.Recall(task='binary', average=None, threshold=self.threshold, num_classes=2),
                  'f1_score_macro': torchmetrics.F1Score(task='binary', average='macro', threshold=self.threshold),
                  'f1_score_asm': torchmetrics.F1Score(task='binary', average=None, threshold=self.threshold, num_classes=2)
            })
            
            self.train_metrics = metrics.clone(prefix='train_')
            self.val_metrics = metrics.clone(prefix='val_')
            self.test_metrics = metrics.clone(prefix='test_')

      def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.Tensor(self.model.forward(x))

      def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], 
                        batch_idx: int) -> torch.Tensor:
            x, y = train_batch
            outputs: torch.Tensor = self(x)
            y = y.unsqueeze(1).type_as(x) # add a channel dimension
            loss: torch.Tensor = self.criterion(outputs, y)
            self.log('train_loss', loss, prog_bar=True, on_step=False,
                  on_epoch=True, sync_dist=True)

            return loss

      def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], 
                          batch_idx: int) -> torch.Tensor:
            x, y = val_batch
            y = y.unsqueeze(1).type_as(x) # add a channel dimension
            outputs: torch.Tensor = self(x)
            loss: torch.Tensor = self.criterion(outputs, y)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > self.threshold).float()
            f1 = self.val_metrics['f1_score_macro'](preds, y)
            
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_f1score', f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            
            return loss

      def test_step(self, test_batch: Tuple[torch.Tensor, torch.Tensor], 
                    batch_idx: int) -> torch.Tensor:
            x, y = test_batch
            y = y.unsqueeze(1).type_as(x) # convert int to float
            logits = self.model(x) # get raw logits
            loss: torch.Tensor = self.criterion(logits, y) # compute loss

            probs = torch.sigmoid(logits) # convert logits to probabilities
            preds = (probs > self.threshold).float() # apply threshold to probrabilities
            metrics = self.test_metrics(preds, y)

            # log loss and accuracy metrics
            self.log('test_loss', loss, sync_dist=True)
            self.log_dict(metrics, sync_dist=True)
    
            # log ASM-specific metrics
            self.log('precision_asm', metrics['precision_asm'][1], sync_dist=True)
            self.log('recall_asm', metrics['recall_asm'][1], sync_dist=True)
            self.log('f1_score_asm', metrics['f1_score_asm'][1], sync_dist=True)

            return loss

      def configure_optimizers(self):
            optimizer = self.optimizer(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                  "scheduler": scheduler,
                  "monitor": "val_loss",
                  "frequency": 1
            }
    }
# mypy: allow-untyped-defs
import torchmetrics
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from enum import Enum
from typing import List, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection

class FusionType(Enum):
    CONCATENATE = "concatenate"
    SUM = "sum"
    AVERAGE = "average"

class LitModelLateFusion(pl.LightningModule):
      """
      PyTorch Lightning module for late fusion of PlanetScope and Sentinel-1 data.
      
      This model implements feature-level fusion of PlanetScope optical imagery and 
      Sentinel-1 SAR data. It uses separate encoders for each data source and fuses
      the extracted features before decoding into the final segmentation output.
      Multiple fusion strategies are supported (concatenation, summation, averaging).
      """
      def __init__(self, 
                  fusion_type: FusionType = FusionType.CONCATENATE,
                  s1_in_channels: int = 3,
                  planet_in_channels: int = 7,
                  lr: float = 1e-3, 
                  threshold: float = 0.5,
                  weight_decay: float = 1e-5,
                  alpha: float = 0.25,
                  gamma: float = 2.0):
            """
            Initialize the Late Fusion model.
            
            Args:
                  fusion_type (FusionType): Strategy for fusing features (concatenate, sum, average)
                  s1_in_channels (int): Number of input channels for Sentinel-1 data
                  planet_in_channels (int): Number of input channels for PlanetScope data
                  lr (float): Learning rate for optimizer
                  threshold (float): Threshold for binary classification
                  weight_decay (float): Weight decay for optimizer
                  alpha (float): Alpha parameter for Focal Loss
                  gamma (float): Gamma parameter for Focal Loss
            """
            super().__init__()
            self.fusion_type = fusion_type
            self.lr = lr
            self.threshold = threshold
            self.weight_decay = weight_decay
            self.alpha = alpha
            self.gamma = gamma
            self.optimizer = torch.optim.Adam
            self.criterion = smp.losses.FocalLoss(alpha=alpha, gamma=gamma, mode='binary')
            
            # init encoders with imagenet weights
            self.s1_encoder = smp.encoders.get_encoder('resnet18', in_channels=s1_in_channels, weights='imagenet')
            self.planet_encoder = smp.encoders.get_encoder('resnet18', in_channels=planet_in_channels, weights='imagenet')
            
            # channel dim depends on fusion type
            if fusion_type == FusionType.CONCATENATE:
                  self.encoder_channels = [128, 128, 256, 512, 1024]
                  self.decoder_channels = [1024, 512, 256, 128, 128]
                  self.channel_adapters = None
            else:
                  # for sum and average, the feature channels won't be doubled
                  self.encoder_channels = [128, 64, 128, 256, 512]  #  do not change first channel dim
                  self.decoder_channels = [1024, 512, 256, 128, 128]  # keep same decoder channels
                  
                  # channel adapters to match expected channel dimensions
                  self.channel_adapters = nn.ModuleList([
                        nn.Conv2d(ch, ch*2, kernel_size=1) 
                        for ch in [64, 128, 256, 512]
                  ])
            
            # init decoder with fixed channel sizes
            self.decoder = UnetDecoder(
                  encoder_channels=self.encoder_channels,
                  decoder_channels=self.decoder_channels,
                  n_blocks=len(self.decoder_channels),
                  use_batchnorm=True,
                  attention_type='scse',
                  center=False
            )
                  
            self.segmentation_head = nn.Sequential(
                  nn.Conv2d(self.decoder_channels[-1], 1, kernel_size=1))
            
            # get the actual channel dimensions from the encoders
            s1_channels = [feat_channels for feat_channels in self.s1_encoder.out_channels[1:]]
            planet_channels = [feat_channels for feat_channels in self.planet_encoder.out_channels[1:]]

            # create normalizers with the correct dimensions for each feature map
            self.s1_normalizers = nn.ModuleList([
                  nn.BatchNorm2d(s1_channels[i]) for i in range(len(s1_channels))
            ])
            self.planet_normalizers = nn.ModuleList([
                  nn.BatchNorm2d(planet_channels[i]) for i in range(len(planet_channels))
            ])
            
            # init metrics
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

      def _fuse_features(self, s1_features: List[torch.Tensor], 
                        planet_features: List[torch.Tensor]) -> List[torch.Tensor]:
            """
            Fuse features from Sentinel-1 and PlanetScope encoders.
            
            Args:
                  s1_features (List[torch.Tensor]): Features extracted from Sentinel-1 data
                  planet_features (List[torch.Tensor]): Features extracted from PlanetScope data
                  
            Returns:
                  List[torch.Tensor]: Fused features using the selected fusion strategy
            """
            if self.fusion_type == FusionType.CONCATENATE:
                  return [torch.cat([s1, planet], dim=1) 
                        for s1, planet in zip(s1_features, planet_features)]
            elif self.fusion_type == FusionType.SUM:
                  return [s1 + planet 
                        for s1, planet in zip(s1_features, planet_features)]
            elif self.fusion_type == FusionType.AVERAGE:
                  return [(s1 + planet) / 2 
                        for s1, planet in zip(s1_features, planet_features)]

      def forward(self, planet_input, s1_input):
            _, _, H, W = planet_input.shape
            
            s1_features = self.s1_encoder(s1_input)
            planet_features = self.planet_encoder(planet_input)
            
            combined_features = []
            for i in range(1, len(s1_features)):
                  s1_norm = self.s1_normalizers[i-1](s1_features[i])
                  planet_norm = self.planet_normalizers[i-1](planet_features[i])
                  
                  # do fusion baed on selected type
                  if self.fusion_type == FusionType.CONCATENATE:
                        fused = torch.cat([s1_norm, planet_norm], dim=1)
                  elif self.fusion_type == FusionType.SUM:
                        fused = s1_norm + planet_norm
                  else:  # AVERAGE
                        fused = (s1_norm + planet_norm) / 2
                        
                  combined_features.append(fused)
            
            x = self.decoder(*combined_features)
            x = self.segmentation_head[0](x)
            x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
            
            return x

      def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                        batch_idx: int) -> torch.Tensor:
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
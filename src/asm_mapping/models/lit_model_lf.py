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
      PyTorch Lightning module for late fusion of PlanetScope and Sentinel-1 data for ASM mapping.
      
      This model implements feature-level fusion of PlanetScope optical imagery and 
      Sentinel-1 SAR data for accurate detection of Artisanal and Small-scale Mining (ASM) sites.
      It uses separate encoders for each data source to extract complementary features:
      - Optical data provides spectral information about land cover
      - SAR data provides textural information and all-weather capabilities
      
      Features from both sources are combined using learnable weights and a selectable
      fusion strategy (concatenation, summation, or averaging) before decoding into
      the final segmentation map. This approach leverages the strengths of both 
      data sources to improve ASM detection performance over single-source models.
      """
      def __init__(self, 
                  fusion_type: FusionType = FusionType.CONCATENATE,
                  s1_in_channels: int = 3,
                  planet_in_channels: int = 6,
                  lr: float = 1e-3, 
                  threshold: float = 0.5,
                  weight_decay: float = 1e-5,
                  alpha: float = 0.25,
                  gamma: float = 2.0):
            """
            Initialize the Late Fusion model with dual-stream architecture.
      
            This model uses two separate ResNet encoders (one for PlanetScope data and 
            one for Sentinel-1 data), with learnable fusion weights to control the relative 
            importance of each data source at different feature levels. The fusion strategy
            (concatenation, sum, or average) determines how features are combined before
            being passed to the U-Net decoder.
            
            Args:
                  fusion_type (FusionType): Strategy for fusing features (concatenate, sum, average)
                  s1_in_channels (int): Number of input channels for Sentinel-1 data (typically 3 for VV, VH, ratio)
                  planet_in_channels (int): Number of input channels for PlanetScope data (typically 6 for RGBNIR+indices)
                  lr (float): Learning rate for Adam optimizer
                  threshold (float): Classification threshold for converting probabilities to binary predictions
                  weight_decay (float): L2 regularization parameter for optimizer
                  alpha (float): Class balancing parameter for Focal Loss (higher values increase weight of positive class)
                  gamma (float): Focusing parameter for Focal Loss (higher values focus more on hard examples)
            
            Note:
                  The model automatically adjusts channel dimensions based on the fusion type selected.
                  Batch normalization is applied to features before fusion to ensure stable training.
                  Metrics tracked include accuracy, precision, recall, and F1-score, with special
                  attention to the ASM class performance.
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
            
            self.fusion_weights = nn.ParameterList([
                  nn.Parameter(torch.FloatTensor([0.7, 0.3])) for _ in range(len(self.s1_encoder.out_channels[1:]))
                  ])
            
            # channel dim depends on fusion type
            if fusion_type == FusionType.CONCATENATE:
                  self.encoder_channels = [128, 128, 256, 512, 1024]
                  self.decoder_channels = [1024, 512, 256, 128, 128]
                  self.channel_adapters = None
            else:
                  # for sum and average, the feature channels won't be doubled
                  self.encoder_channels = [128, 64, 128, 256, 512] 
                  self.decoder_channels = [1024, 512, 256, 128, 128]
                  
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
                  
                  # get learned weights
                  weights = torch.softmax(self.fusion_weights[i-1], dim=0)
                  
                  # apply weigthed fusion (based on fusion type)
                  if self.fusion_type == FusionType.CONCATENATE:
                        # apply weights before concatenation
                        fused = torch.cat([s1_norm * weights[0], planet_norm * weights[1]], dim=1)
                  elif self.fusion_type == FusionType.SUM:
                        fused = s1_norm * weights[0] + planet_norm * weights[1]
                  else:  # AVERAGE
                        fused = (s1_norm * weights[0] + planet_norm * weights[1])
                        
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
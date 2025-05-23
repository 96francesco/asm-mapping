import logging
from pathlib import Path
from torch.utils.data import random_split
import os
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from asm_mapping.models.lit_model_lf import FusionType
from asm_mapping.data.fusion_dataset import ResampleStrategy
import torch

from asm_mapping.data import DatasetMode
from asm_mapping.train_test_predict.utils import get_dataset, get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_test_split(config, split_n):
      # set paths
      if config["mode"] == "standalone":
            config["training_dir"] = os.path.join(
                  config["base_data_dir"], f"split_{split_n}", "training_set"
            )
            config["testing_dir"] = os.path.join(
                  config["base_data_dir"], f"split_{split_n}", "testing_set"
            )
      else:  # fusion modes
            config["training_dir"] = config["base_data_dir"]
            config["testing_dir"] = config["base_data_dir"]

      # get model and dataset classes
      dataset_class = get_dataset(config)
      model_class = get_model(config)

      # initialize datasets
      dataset_mode = DatasetMode[config["dataset_mode"]]
      if config["mode"] == "standalone":
            train_dataset = dataset_class(
                  config["training_dir"],
                  mode=dataset_mode,
                  transforms=config["augmentation"],
                  pad=config["pad"],
                  split=f"split_{split_n}"
                  )
      else: 
            train_dataset = dataset_class(
                  config["training_dir"],
                  mode=dataset_mode,
                  transforms=config["augmentation"],
                  pad=config["pad"],
                  split=split_n, 
                  is_test=False,
                  resample_strategy=ResampleStrategy[config["resample_strategy"]]
            )

      # extract validation set from training set
      train_size = int(0.8 * len(train_dataset))
      val_size = len(train_dataset) - train_size
      train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
      
      if config["mode"] == "standalone":
            test_dataset = dataset_class(
                  config["testing_dir"],
                  mode=dataset_mode,
                  transforms=False,
                  pad=config["pad"],
                  split=f"split_{split_n}"
            )
      else:  # fusion modes
            test_dataset = dataset_class(
                  config["testing_dir"],
                  mode=dataset_mode,
                  transforms=False,
                  pad=config["pad"],
                  split=split_n,
                  is_test=True,
                  resample_strategy=ResampleStrategy[config["resample_strategy"]],
            )

      # initialize dataloaders
      train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            persistent_workers=config["persistent_workers"],
            pin_memory=config["pin_memory"],
            prefetch_factor=config["prefetch_factor"],
      )
      val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            persistent_workers=config["persistent_workers"],
            pin_memory=config["pin_memory"],
            prefetch_factor=config["prefetch_factor"],
      )
      test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            persistent_workers=config["persistent_workers"],
            pin_memory=config["pin_memory"],
            prefetch_factor=config["prefetch_factor"],
      )

      # initialize model
      unet = smp.Unet(
            encoder_name=config["encoder_name"],
            decoder_attention_type=config["decoder_attention_type"],
            decoder_use_batchnorm=config["decoder_use_batchnorm"],
            in_channels=config["in_channels"],
            classes=1,
      )
      if config['mode'] == 'standalone':
            model = model_class(
                  model=unet,
                  lr=config["learning_rate"],
                  weight_decay=config["weight_decay"],
                  threshold=config["threshold"],
                  alpha=config["alpha"],
                  gamma=config["gamma"],
            )
      elif config['mode'] == 'late_fusion':
            model = model_class(
            fusion_type=FusionType[config["fusion_type"]],
            s1_in_channels=config["s1_in_channels"],
            planet_in_channels=config["planet_in_channels"],
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            threshold=config["threshold"],
            alpha=config["alpha"],
            gamma=config["gamma"]
            )
      elif config['mode'] == 'early_fusion':
            model = model_class(
            s1_in_channels=config["s1_in_channels"],
            planet_in_channels=config["planet_in_channels"],
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            threshold=config["threshold"],
            alpha=config["alpha"],
            gamma=config["gamma"]
            )

      # set callbacks
      callbacks = [
            EarlyStopping(
                  monitor=config["early_stopping"]["monitor"],
                  patience=config["early_stopping"]["patience"],
                  mode=config["early_stopping"]["mode"],
            ),
            ModelCheckpoint(
                  dirpath=Path(config["checkpoint_dir"]) / f"split_{split_n}",
                  filename=f"{config['experiment_name']}_split_{split_n}"
                  + "_{epoch:02d}_{val_f1_score_asm:.3f}",
                  monitor=config["early_stopping"]["monitor"],
                  mode=config["early_stopping"]["mode"],
                  save_top_k=1,
            ),
      ]

      # logger
      tb_logger = TensorBoardLogger(
            save_dir=config["log_dir"], name=f"{config['experiment_name']}_split_{split_n}"
      )

      # train model
      trainer = pl.Trainer(
            max_epochs=config["epochs"],
            callbacks=callbacks,
            logger=tb_logger,
            devices=config["gpus"],
            log_every_n_steps=10,
            gradient_clip_val=0.5,
      )
      trainer.fit(model, train_loader, val_loader)
      
      # save model
      checkpoint_dir = Path(config["checkpoint_dir"]) / f"split_{split_n}"
      model_name = f"{config['experiment_name']}_split_{split_n}"
      pth_path = checkpoint_dir / f"{model_name}.pth"
      torch.save(model.state_dict(), pth_path)

      # test model
      results = trainer.test(model, test_loader)

      return results[0] if results else {}

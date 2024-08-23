# mypy: allow-untyped-defs
import logging
import argparse

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from asm_mapping.data import DatasetMode
from asm_mapping.train_test_predict.utils import load_config, get_dataset, get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
      parser = argparse.ArgumentParser(description="Test ASM segmentation model")
      parser.add_argument("--config", type=str, required=True, help="Path to config file")
      args = parser.parse_args()
      
      # load config
      config = load_config(args.config)
      
      # set seed
      pl.seed_everything(config['seed'])

      # get dataset and model classes
      dataset_class = get_dataset(config)
      model_class = get_model(config)

      # dataset initialization
      dataset_mode = DatasetMode[config['dataset_mode']]
      test_dataset = dataset_class(
            config['test_dir'], 
            mode=dataset_mode,
            transforms=False,
            pad=config['pad']
      )
      
      # dataloader initialization
      test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers'],
            persistent_workers=config['persistent_workers'],
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor']
      )

      # model initialization
      unet = smp.Unet(
            encoder_name=config['encoder_name'],
            decoder_attention_type=config['decoder_attention_type'],
            decoder_use_batchnorm=config['decoder_use_batchnorm'],
            in_channels=config['in_channels'],
            classes=1
      )
      model = model_class(
            model=unet,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            threshold=config['threshold'],
            alpha=config['alpha'],
            gamma=config['gamma']
      )
      model.load_state_dict(torch.load(config['model_path']))

      # Trainer
      trainer = pl.Trainer(
            devices=config['gpus']
      )

      # start testing
      trainer.test(model, test_loader)

if __name__ == "__main__":
      main()
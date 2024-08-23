# mypy: allow-untyped-defs
import logging
import argparse

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp # type: ignore
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from asm_mapping.data import DatasetMode
from asm_mapping.train_test_predict.utils import load_config, get_dataset, get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASM segmentation model")
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
    train_dataset = dataset_class(
        config['training_dir'], 
        mode=dataset_mode,
        transforms=config['augmentation']['enabled'],
        pad=config['pad']
    )
    
    # extract validation subset from training dataset
    train_size = int(config['train_val_split'] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # dataloader initialization
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=config['train_loader_shuffle'], 
        num_workers=config['num_workers'],
        persistent_workers=config['persistent_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=config['val_loader_shuffle'], 
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

    # callbacks
    early_stop = EarlyStopping(
        monitor=config['early_stopping']['monitor'],
        patience=config['early_stopping']['patience'],
        mode=config['early_stopping']['mode']
    )
    checkpoint = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename=config['experiment_name'] + "-{epoch:02d}-{val_f1score:.2f}"
    )

    # TB logger
    tb_logger = TensorBoardLogger(
        save_dir=config['log_dir'], 
        name=config['experiment_name']
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        callbacks=[early_stop, checkpoint],
        val_check_interval=config['val_check_interval'],
        logger=tb_logger,
        devices=config['gpus']
        # TODO: log_every_n_steps, define accelerator and DDP strategy?
    )

    # start training
    trainer.fit(model, train_loader, val_loader)
    
    # save model
    torch.save(model.state_dict(), f"{config['checkpoint_dir']}/{config['experiment_name']}_final.pth")

if __name__ == "__main__":
    main()
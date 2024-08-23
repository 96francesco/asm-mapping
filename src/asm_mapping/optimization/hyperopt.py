# mypy: allow-untyped-defs
import optuna
import argparse
import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp

from typing import Dict, Any
from optuna.trial import Trial
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from asm_mapping.data import DatasetMode
from asm_mapping.train_test_predict.utils import load_config, get_dataset, get_model

def objective(trial: Trial, train_set: Dataset, val_set: Dataset, # type: ignore
              config: Dict[str, Any]) -> float:
    seed_everything(config['seed'], workers=True)

    # hyperparameters to optimize
    weight_decay = trial.suggest_float('weight_decay', *config['optimization']['weight_decay'], log=True)
    batch_size = trial.suggest_categorical('batch_size', config['data'].get('batch_size', [8, 16, 32])) 
    lr = trial.suggest_float("lr", *config['optimization']['learning_rate'], log=True)
    alpha = trial.suggest_float("alpha", *config['optimization']['alpha'], step=0.05)
    gamma = trial.suggest_float("gamma", *config['optimization']['gamma'], step=1.0)
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    attention_type = trial.suggest_categorical('attention_type', [None, 'scse'])
    encoder_name = trial.suggest_categorical('encoder_name', config['optimization']['encoder_name'])
    
    # initialize dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=config['data']['num_workers'],
                              persistent_workers=True, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                            num_workers=config['data']['num_workers'],
                            persistent_workers=True, pin_memory=True, prefetch_factor=2)
    
    # initialize model
    model_class = get_model(config)
    unet = smp.Unet(
        encoder_name=encoder_name,
        decoder_attention_type=attention_type,
        in_channels=config['model']['in_channels'],
        classes=1,
        decoder_use_batchnorm=use_batch_norm
    )
    model = model_class(
        model=unet,
        lr=lr,
        weight_decay=weight_decay,
        threshold=config['model']['threshold'],
        alpha=alpha,
        gamma=gamma
    )

    # initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config['model']['epochs'],
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=True,
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    # fit model and return F1 score
    trainer.fit(model, train_loader, val_loader)
    val_f1score = trainer.callback_metrics["val_f1score"].item()

    return val_f1score

def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for ASM mapping model")
    parser.add_argument("--config", type=str, default="hyperopt_config.yaml", 
                        help="Path to hyperopt config file")
    args = parser.parse_args()

    torch.cuda.empty_cache()

    config = load_config(args.config)
    seed_everything(config['seed'], workers=True)
    
    # initialize dataset and dataloaders
    dataset_class = get_dataset(config)
    dataset_mode = DatasetMode[config['dataset_mode']]
    training_dataset = dataset_class(
        config['data']['training_dir'],
        mode=dataset_mode,
        transforms=True
    )
    train_size = int((1 - config['validation_split']) * len(training_dataset))
    val_size = len(training_dataset) - train_size
    train_set, val_set = random_split(training_dataset, [train_size, val_size])
   
    storage = optuna.storages.RDBStorage(config['database_url'])
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=config['seed']),
        pruner=pruner,
        storage=storage,
        study_name=config['study_name'],
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, train_set, val_set, config),
                   n_trials=config['n_trials'], gc_after_trial=True)

if __name__ == "__main__":
    main()
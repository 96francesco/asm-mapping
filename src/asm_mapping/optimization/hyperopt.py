# mypy: allow-untyped-defs
import optuna
import argparse
import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp

from typing import Dict, Any
from optuna.trial import Trial
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import seed_everything
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from asm_mapping.data import DatasetMode
from asm_mapping.data.fusion_dataset import ResampleStrategy
from asm_mapping.train_test_predict.utils import load_config, get_dataset, get_model
from asm_mapping.models.lit_model_lf import FusionType

def objective(trial: Trial, config: Dict[str, Any]) -> float:
      seed_everything(config['seed'], workers=True)

      # hyperparameters to optimize
      weight_decay = trial.suggest_float('weight_decay', *config['optimization']['weight_decay'], log=True)
      batch_size = trial.suggest_categorical('batch_size', config['data'].get('batch_size', [8, 16, 32])) 
      lr = trial.suggest_float("lr", *config['optimization']['learning_rate'], log=True)
      alpha = trial.suggest_float("alpha", *config['optimization']['alpha'], step=0.05)
      gamma = trial.suggest_float("gamma", *config['optimization']['gamma'], step=1.0)
      encoder_name = trial.suggest_categorical('encoder_name', config['optimization']['encoder_name'])
      resample_strategy_str = trial.suggest_categorical('resample_strategy', config['fusion']['resample_strategy'])
      resample_strategy = ResampleStrategy[resample_strategy_str]
      
      # initialize dataset
      dataset_class = get_dataset(config)
      dataset_mode = DatasetMode[config['dataset_mode']]
      
      split_n = 0
      training_dataset = dataset_class(
            config['data']['training_dir'],
            split=split_n,
            mode=dataset_mode,
            transforms=True,
            pad=config['data']['pad'],
            resample_strategy=resample_strategy 
      )
      
      # split dataset
      train_size = int((1 - config['validation_split']) * len(training_dataset))
      val_size = len(training_dataset) - train_size
      train_set, val_set = random_split(training_dataset, [train_size, val_size])
      
      # init dataloaders
      train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                    num_workers=config['data']['num_workers'],
                                    persistent_workers=True, pin_memory=True, prefetch_factor=2)
      val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                              num_workers=config['data']['num_workers'],
                              persistent_workers=True, pin_memory=True, prefetch_factor=2)
      
      # init model
      model_class = get_model(config)
      if config['mode'] == 'standalone':
            unet = smp.Unet(
                  encoder_name=encoder_name,
                  decoder_attention_type='scse',
                  in_channels=config['model']['in_channels'],
                  classes=1,
                  decoder_use_batchnorm=True
            )
            model = model_class(
                  model=unet,
                  lr=lr,
                  weight_decay=weight_decay,
                  threshold=config['model']['threshold'],
                  alpha=alpha,
                  gamma=gamma
            )
      elif config['mode'] == 'late_fusion':
            fusion_type_str = trial.suggest_categorical('fusion_type', config['fusion']['type'])
            initial_weights = config['fusion']['initial_weights']
            model = model_class(
                  fusion_type=FusionType[fusion_type_str],
                  s1_in_channels=config['fusion']['s1_in_channels'],
                  planet_in_channels=config['fusion']['planet_in_channels'],
                  lr=lr,
                  weight_decay=weight_decay,
                  threshold=config['model']['threshold'],
                  alpha=alpha,
                  gamma=gamma
            )
            
            # set fixed initial weights for fusion (after model initialization)
            for i in range(len(model.fusion_weights)):
                  model.fusion_weights[i].data = torch.tensor(initial_weights, dtype=torch.float32)
      
      elif config['mode'] == 'early_fusion':
            model = model_class(
                  s1_in_channels=config['fusion']['s1_in_channels'],
                  planet_in_channels=config['fusion']['planet_in_channels'],
                  lr=lr,
                  weight_decay=weight_decay,
                  threshold=config['model']['threshold'],
                  alpha=alpha,
                  gamma=gamma
            )
      else:
            raise ValueError(f"Unsupported model mode: {config['mode']}")

      # init Trainer
      trainer = pl.Trainer(
            max_epochs=config['model']['epochs'],
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            accelerator=config['trainer']['accelerator'],
            devices=config['trainer']['devices'],
            strategy='auto',
      )

      # train model
      trainer.fit(model, train_loader, val_loader)
      
      try:
            val_f1score = trainer.callback_metrics["val_f1_score_asm"].item()
      except:
            val_f1score = trainer.callback_metrics.get("val_f1_score_asm", 0.0)
            if hasattr(val_f1score, 'item'):
                  val_f1score = val_f1score.item()
      
      return val_f1score

def main() -> None:
      parser = argparse.ArgumentParser(description="Hyperparameter optimization for ASM mapping model")
      parser.add_argument("--config", type=str, default="hyperopt_config.yaml", 
                              help="Path to hyperopt config file")
      args = parser.parse_args()

      torch.cuda.empty_cache()

      config = load_config(args.config)
      seed_everything(config['seed'], workers=True)
      
      # init Optuna study
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

      # run optimization
      study.optimize(lambda trial: objective(trial, config),
                        n_trials=config['n_trials'], gc_after_trial=True)
      print("\nBest trial:")
      trial = study.best_trial
      print(f"  Value: {trial.value}")
      print("  Params:")
      for key, value in trial.params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
      main()
import os

import yaml # type: ignore
import pytest
import torch
import pytorch_lightning as pl

from asm_mapping.models.lit_model_standalone import LitModelStandalone

# load config
config_path = os.path.join(os.path.dirname(__file__), 'tests_config.yaml')
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
    
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CHANNELS = config['model']['in_channels']

@pytest.fixture
def lit_model():
    model = LitModelStandalone(in_channels=IN_CHANNELS)
    return model

@pytest.fixture
def input_tensor(lit_model):
      return torch.rand((1, config['model']['in_channels'], 256, 256))

def test_model_init(lit_model):
      assert isinstance(lit_model, pl.LightningModule)
      assert lit_model.weight_decay == 1e-5
      assert lit_model.lr == 1e-3
      assert lit_model.threshold == 0.5
      assert lit_model.alpha == 0.25
      assert lit_model.gamma == 2.0
      assert lit_model.in_channels == config['model']['in_channels']

def test_forward(lit_model, input_tensor):
      output = lit_model(input_tensor)
      assert output.shape == torch.Size([1, 1, 256, 256])

def test_training_step(lit_model, input_tensor):
      batch = (input_tensor, torch.randint(0, 2, (1, 256, 256)))
      loss = lit_model.training_step(batch, 0)
      assert isinstance(loss, torch.Tensor)
      assert loss.shape == torch.Size([])
      
def test_validation_step(lit_model, input_tensor):
      batch = (input_tensor, torch.randint(0, 2, (1, 256, 256)))
      loss = lit_model.validation_step(batch, 0)
      assert isinstance(loss, torch.Tensor)
      assert loss.shape == torch.Size([])

def test_test_step(lit_model, input_tensor):
      batch = (input_tensor, torch.randint(0, 2, (1, 256, 256)))
      loss = lit_model.test_step(batch, 0)
      assert isinstance(loss, torch.Tensor)
      assert loss.shape == torch.Size([])

def test_configure_optimizers(lit_model):
      optimizer = lit_model.configure_optimizers()
      assert isinstance(optimizer, dict)
      assert "optimizer" in optimizer
      assert "lr_scheduler" in optimizer
      assert isinstance(optimizer["optimizer"], torch.optim.Adam)
      
      lr_scheduler = optimizer["lr_scheduler"]
      assert isinstance(lr_scheduler, dict)
      assert "scheduler" in lr_scheduler
      assert isinstance(lr_scheduler["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
      assert lr_scheduler["monitor"] == "val_loss"
      assert lr_scheduler["frequency"] == 1
      
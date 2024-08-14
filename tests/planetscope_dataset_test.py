import os

import yaml # type: ignore
import pytest
import torch
import numpy as np

from asm_mapping.data.planetscope_dataset import PlanetScopeDataset, DatasetMode

# load config
config_path = os.path.join(os.path.dirname(__file__), 'tests_config.yaml')
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
    
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, config['test_data']['dir'])
TEST_DATA_LEN = config['test_data']['len']

@pytest.fixture
def planetscope_dataset():
    dataset = PlanetScopeDataset(data_dir=TEST_DATA_DIR, mode=DatasetMode.STANDALONE)
    return dataset

def test_dataset_init(planetscope_dataset):
      assert isinstance(planetscope_dataset, PlanetScopeDataset)
      assert planetscope_dataset.data_dir == TEST_DATA_DIR
      assert planetscope_dataset.mode == DatasetMode.STANDALONE
      assert not planetscope_dataset.pad
      assert planetscope_dataset.transforms is None

def test_dataset_len(planetscope_dataset):
      assert len(planetscope_dataset) == TEST_DATA_LEN

def test_getitem(planetscope_dataset):
      item = planetscope_dataset[0]
      assert isinstance(item, tuple)
      assert len(item) == 2
      img, mask = item
      assert isinstance(img, torch.Tensor)
      assert isinstance(mask, torch.Tensor)
      assert img.shape[0] == 6 # 4 bands + 2 VIs
      assert mask.shape == img.shape[1:]
      
def test_ndvi():
      dummy_img = np.random.rand(4, 10, 10).astype(np.float32)
      ndvi = PlanetScopeDataset.ndvi(dummy_img)
      assert isinstance(ndvi, torch.Tensor)
      assert ndvi.shape == (1, 10, 10)

def test_ndwi():
      dummy_img = np.random.rand(4, 10, 10).astype(np.float32)
      ndwi = PlanetScopeDataset.ndwi(dummy_img)
      assert isinstance(ndwi, torch.Tensor)
      assert ndwi.shape == (1, 10, 10)

def test_normalization(planetscope_dataset):
      dummy_img = np.random.rand(4, 10, 10).astype(np.float32)
      normalized_img = PlanetScopeDataset.normalize_global_percentile(planetscope_dataset,
                                                                      image=dummy_img)
      assert normalized_img.shape == dummy_img.shape
      assert normalized_img.min() >= 0 and normalized_img.max() <= 1

def test_handle_nan_values(planetscope_dataset):
      dummy_img = np.random.rand(4, 10, 10).astype(np.float32)
      dummy_img[0, 0, 0] = np.nan
      handled = PlanetScopeDataset.handle_nan_values(planetscope_dataset, 
                                                     img=dummy_img)
      assert not np.isnan(handled).any()

def test_dynamic_pad():
      dummy_tensor = torch.rand(4, 30, 30)
      padded = PlanetScopeDataset.dynamic_pad(dummy_tensor)
      assert padded.shape[1] % 32 == 0 and padded.shape[2] % 32 == 0
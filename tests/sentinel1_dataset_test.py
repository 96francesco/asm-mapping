# mypy: allow-untyped-defs
import os

import yaml # type: ignore
import pytest
import torch
import numpy as np

from asm_mapping.data.sentinel1_dataset import Sentinel1Dataset
from asm_mapping.data.dataset_mode import DatasetMode

# load config
config_path = os.path.join(os.path.dirname(__file__), 'tests_config.yaml')
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
    
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, config['test_data']['dir'])
TEST_DATA_LEN = config['test_data']['len']

@pytest.fixture
def s1_dataset():
    dataset = Sentinel1Dataset(data_dir=TEST_DATA_DIR, mode=DatasetMode.STANDALONE)
    return dataset

def test_dataset_init(s1_dataset):
      assert isinstance(s1_dataset, Sentinel1Dataset)
      assert s1_dataset.data_dir == TEST_DATA_DIR
      assert s1_dataset.mode == DatasetMode.STANDALONE
      assert not s1_dataset.pad
      assert s1_dataset.transforms is None

def test_dataset_len(s1_dataset):
      assert len(s1_dataset) == TEST_DATA_LEN

def test_vv_vh_ratio(s1_dataset):
    dummy_img = np.random.rand(2, 10, 10).astype(np.float32)
    ratio = Sentinel1Dataset.vv_vh_ratio(dummy_img)
    assert ratio.shape == (1, 10, 10)
    assert np.allclose(ratio[0], dummy_img[0] - dummy_img[1])

def test_getitem(s1_dataset):
      item = s1_dataset[0]
      assert isinstance(item, tuple)
      assert len(item) == 2
      img, mask = item
      assert isinstance(img, torch.Tensor)
      assert isinstance(mask, torch.Tensor)
      assert img.shape[0] == 3 # VV + VH + VV/VH
      assert mask.shape == img.shape[1:]
      
def test_normalization(s1_dataset):
      dummy_img = np.random.rand(4, 10, 10).astype(np.float32)
      normalized_img = Sentinel1Dataset.normalize_global_percentile(s1_dataset,
                                                                      image=dummy_img)
      assert normalized_img.shape == dummy_img.shape
      assert normalized_img.min() >= 0 and normalized_img.max() <= 1

def test_handle_nan_values(s1_dataset):
      dummy_img = np.random.rand(4, 10, 10).astype(np.float32)
      dummy_img[0, 0, 0] = np.nan
      handled = Sentinel1Dataset.handle_nan_values(s1_dataset, 
                                                     img=dummy_img)
      assert not np.isnan(handled).any()

def test_dynamic_pad():
      dummy_tensor = torch.rand(4, 30, 30)
      padded = Sentinel1Dataset.dynamic_pad(dummy_tensor)
      assert padded.shape[1] % 32 == 0 and padded.shape[2] % 32 == 0
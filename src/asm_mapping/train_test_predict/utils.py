import yaml # type: ignore
import logging
from typing import Dict, Any

from asm_mapping.data.planetscope_dataset import PlanetScopeDataset
from asm_mapping.data.sentinel1_dataset import Sentinel1Dataset
from asm_mapping.data.fusion_dataset import FusionDataset
from asm_mapping.models import LitModelStandalone, LitModelLateFusion, LitModelEarlyFusion

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Any:
      try:
            with open(config_path, 'r') as f:
                  return yaml.safe_load(f)
      except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

def get_dataset(config: Dict[str, Any]) -> Any:
      dataset_map = {
            "PlanetScope": PlanetScopeDataset,
            "Sentinel-1": Sentinel1Dataset,
            "Fusion" :  FusionDataset,
      }
      try:
            return dataset_map[config['data_source']]
      except KeyError:
            logger.error(f"Invalid data source: {config['data_source']}")
            raise

def get_model(config: Dict[str, Any]) -> Any:
      model_map = {
            "standalone": LitModelStandalone,
            "late_fusion": LitModelLateFusion,
            "early_fusion": LitModelEarlyFusion,
            # TODO: add other model classes when implemented
      }
      try:
            return model_map[config['mode']]
      except KeyError:
            logger.error(f"Invalid mode: {config['mode']}")
            raise
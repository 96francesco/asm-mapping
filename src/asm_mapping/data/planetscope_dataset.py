# mypy: allow-untyped-defs
import os
import torch
import rasterio # type: ignore
import numpy as np
import random

from typing import List, Optional, Tuple, Dict, Any, Union
from torchvision import transforms as T # type: ignore
from scipy.ndimage import median_filter # type: ignore
from torch.utils.data import Dataset
from PIL import Image

from asm_mapping.data.dataset_mode import DatasetMode

class PlanetScopeDataset(Dataset): # type: ignore
      """
      @TODO fix the docstring
      """
      def __init__(self, data_dir: str, mode: DatasetMode = DatasetMode.STANDALONE,
                   pad: bool = False, transforms: bool = False):
            self.data_dir: str = data_dir
            self.mode: DatasetMode = mode
            self.pad: bool = pad
            self.transforms: Optional[T.Compose] = None
            
            self.config: Dict[DatasetMode, Dict[str, Any]] = self._get_mode_config()
            self.img_folder, self.gt_folder = self._setup_folders()
            self.dataset: List[Union[str, Tuple[str, str]]] = self._create_dataset()

            if transforms:
                  self.transforms  = T.Compose([
                  T.RandomHorizontalFlip(),
                  T.RandomVerticalFlip(),
            ])
            
            self.percentiles: Tuple[float, float] = self.compute_global_percentiles()
      
      def _get_mode_config(self) -> Dict[DatasetMode, Dict[str, Any]]:
            return {
                  DatasetMode.STANDALONE: {
                        'img_subfolder': 'images',
                        'gt_subfolder': 'gt',
                        'use_gt': True
                  },
                  DatasetMode.FUSION: {
                        'img_subfolder': 'images/planet',
                        'gt_subfolder': 'gt',
                        'use_gt': True
                  },
                  DatasetMode.INFERENCE: {
                        'img_subfolder': '',
                        'gt_subfolder': None,
                        'use_gt': False
                  }
            }
            
      def _setup_folders(self) -> Tuple[str, Optional[str]]:
            config = self.config[self.mode]
            img_folder = os.path.join(self.data_dir, config['img_subfolder'])
            gt_folder = os.path.join(self.data_dir, config['gt_subfolder']) if config['gt_subfolder'] else None
            return img_folder, gt_folder if gt_folder is not None else None
      
      def _create_dataset(self) -> List[Union[str, Tuple[str, str]]]:
            dataset: List[Union[str, Tuple[str, str]]] = []
            img_filenames = sorted(os.listdir(self.img_folder))
            
            if self.config[self.mode]['use_gt']:
                  if self.gt_folder is not None:
                        gt_filenames = sorted(os.listdir(self.gt_folder))
                        for img_name in img_filenames:
                              gt_name = 'nicfi_gt_' + img_name.split('_')[-1]
                              if gt_name in gt_filenames:
                                    img_path = os.path.join(self.img_folder, img_name)
                                    gt_path = os.path.join(self.gt_folder, gt_name)
                                    dataset.append((img_path, gt_path))
                  else:
                        raise ValueError("Ground truth folder is None but use_gt is True")
            else:
                  dataset = [os.path.join(self.img_folder, img_name) for img_name in img_filenames]
            
            return dataset

      @staticmethod
      def ndvi(img: np.ndarray[Any, Any]) -> torch.Tensor:
            red = img[2, :, :]
            nir = img[3, :, :]
            ndvi = (nir - red) / (nir + red + 1e-10)  # small epsilon to avoid division by zero

            # convert to tensor and add channel dimension [1, height, width]
            return torch.from_numpy(ndvi).unsqueeze(0)

      @staticmethod
      def ndwi(img: np.ndarray[Any, Any]) -> torch.Tensor:
            green = img[1, :, :]
            nir = img[3, :, :]
            ndwi = (green - nir) / (green + nir + 1e-10)

            return torch.from_numpy(ndwi).unsqueeze(0)
      

      def __len__(self) -> int:
            return len(self.dataset)

      def compute_global_percentiles(self) -> Tuple[float, float]:
            all_values = []
            for i in random.sample(self.dataset, min(100, len(self.dataset))):
                  # check if i is a tuple (with GT) or a string (without GT)
                  if isinstance(i, tuple):
                        img_path, _ = i
                  else:
                        img_path = None

                  with rasterio.open(img_path, 'r') as ds:
                        img = ds.read().astype(np.float32)
                  img = self.handle_nan_values(img)
                  all_values.append(img)
            
            all_values = np.concatenate([arr.flatten() for arr in all_values])
            
            p2 = np.percentile(all_values, 2)
            p98 = np.percentile(all_values, 98)
            
            # print(f"Global stats - min: {all_values.min()}, 
            #       max: {all_values.max()}, 
            #       mean: {all_values.mean()}")
            # print(f"Computed percentiles - p2: {p2}, p98: {p98}")
            
            return p2, p98
      
      def normalize_global_percentile(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            p2, p98 = self.percentiles
            
            normalized = (image - p2) * (1 / (p98 - p2))
            
            normalized = np.clip(normalized, 0, 1)

            return normalized
      
      def handle_nan_values(self, img: np.ndarray[Any, Any]) -> Any:
            img = np.nan_to_num(img, nan=np.nanmedian(img))
            return median_filter(img, size=3) 

      @staticmethod
      def dynamic_pad(image: torch.Tensor, multiple: int = 32) -> torch.Tensor:
            # @TODO: fix docstring
            """_summary_

            Args:
                image (_type_): _description_
                multiple (int, optional): _description_. Defaults to 32.

            Returns:
                _type_: _description_
            """
            if len(image.shape) == 3:
                  _, h, w = image.shape
            else:
                  h, w = image.shape
            pad_h = (multiple - h % multiple) % multiple
            pad_w = (multiple - w % multiple) % multiple
            
            return torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), 
                                           mode='constant', 
                                           value=0)

      def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:                  
            item = self.dataset[idx]
            if isinstance(item, tuple):
                  img_path, gt_path = item
            else:
                  img_path = item
                  gt_path = None
                  
            # read and process image
            # conversiong to float32 is done to ensure correct normalization
            with rasterio.open(img_path, 'r') as ds:
                  img = ds.read().astype(np.float32)

            # handle NaN values
            img = self.handle_nan_values(img)

            # compute vegetation indices
            ndvi = self.ndvi(img)
            ndwi = self.ndwi(img)
            
            # normalize image
            img = self.normalize_global_percentile(img)

            # convert image to tensor
            img_tensor = torch.from_numpy(img).float()

            # stack NDVI and ndwi as additional channels
            img_tensor = torch.cat((img_tensor, ndvi, ndwi), dim=0)

            # apply data augmentation
            if self.transforms:
                  random.seed(96)
                  torch.manual_seed(69)
                  img_tensor = self.transforms(img_tensor)

            # apply dynamic padding
            if self.pad:
                  img_tensor = self.dynamic_pad(img_tensor)
            
            # if not self.is_inference:
            if self.config[self.mode]['use_gt']:
                  gt_path = self.dataset[idx][1]
                  gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32)
                  gt_tensor = torch.from_numpy(gt).long()
                  
                  if self.transforms:
                        gt_tensor = gt_tensor.unsqueeze(0)
                        random.seed(96)
                        torch.manual_seed(69)
                        gt_tensor = self.transforms(gt_tensor)
                        gt_tensor = gt_tensor.squeeze(0)
                  
                  # apply dynamic padding
                  if self.pad:
                        gt_tensor = self.dynamic_pad(gt_tensor)
                        
                  return img_tensor, gt_tensor
            else:
                  return img_tensor
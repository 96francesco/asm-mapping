from enum import Enum
import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
from typing import Tuple, Optional

from asm_mapping.data.dataset_mode import DatasetMode

class ResampleStrategy(Enum):
      UPSAMPLE_S1 = "upsample_s1"  
      DOWNSAMPLE_PLANET = "downsample_planet"

class FusionDataset(Dataset):
      """
      Dataset class for handling the fusion of PlanetScope and Sentinel-1 data.
    
      This class loads and processes pairs of PlanetScope optical imagery and Sentinel-1 
      SAR data for fusion-based deep learning models. It handles data loading, 
      resampling between different spatial resolutions, and data augmentation.
      """
      def __init__(self, 
                data_dir: str,
                split: int,
                mode: DatasetMode = DatasetMode.FUSION,
                transforms: bool = False,
                resample_strategy: ResampleStrategy = ResampleStrategy.DOWNSAMPLE_PLANET,
                pad: bool = False):
            """
            Initialize the fusion dataset.
        
            Args:
                  data_dir (str): Base directory for the dataset
                  split (int): Split number to use (e.g., 0, 1, 2...)
                  mode (DatasetMode): Mode to operate in (standalone, fusion, inference)
                  transforms (bool): Whether to apply data augmentation
                  resample_strategy (ResampleStrategy): Strategy for resampling between different resolutions
                  pad (bool): Whether to pad images to be divisible by 32
            """
        
            self.data_dir = data_dir
            self.split = split
            self.mode = mode
            self.resample_strategy = resample_strategy
            self.pad = pad
            self.transforms = None
            
            if transforms:
                  self.transforms = T.Compose([
                  T.RandomHorizontalFlip(),
                  T.RandomVerticalFlip(),
                  T.RandomRotation(degrees=90),
                  T.RandomAffine(degrees=0, scale=(0.9, 1.1), shear=None)
                  ])
                  
            # setup paths
            self.planet_dir = os.path.join(data_dir, 'ps_split', f'split_{split}', 'training_set')
            self.s1_dir = os.path.join(data_dir, 's1_split', f'split_{split}', 'training_set')
            self.gt_dir = os.path.join(self.planet_dir, 'masks')
            
            # get matched pairs of files
            self.file_pairs = self._get_matched_files()
            
      def _get_matched_files(self) -> list:
            """
            Find all matching file triplets (Planet, S1, GT).
            
            Returns:
                  list: List of tuples containing paths to matching (Planet, S1, GT) file triplets
            """
            planet_files = sorted(os.listdir(os.path.join(self.planet_dir, 'images')))
            s1_files = sorted(os.listdir(os.path.join(self.s1_dir, 'images')))
            gt_files = sorted(os.listdir(os.path.join(self.planet_dir, 'masks')))
            
            pairs = []
            
            for p_file in planet_files:
                  # extract the index from the filename
                  index = p_file.split('_')[-1]
                  s1_file = f'img_{index}'
                  gt_file = f'mask_{index}'
                  
                  # check if corresponding S1 and GT files exist
                  if (s1_file in s1_files and 
                        gt_file in gt_files):
                        pairs.append((
                        os.path.join('images', p_file), 
                        os.path.join(self.s1_dir, 'images', s1_file), 
                        os.path.join('masks', gt_file)
                        ))
            
            return pairs

      def _resample_data(self, planet_data: np.ndarray, 
                        s1_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Resample data according to chosen strategy.
            
            Either upsamples Sentinel-1 data to match PlanetScope resolution
            or downsamples PlanetScope data to match Sentinel-1 resolution.
            
            Args:
                  planet_data (np.ndarray): PlanetScope multispectral data
                  s1_data (np.ndarray): Sentinel-1 SAR data
                  
            Returns:
                  Tuple[np.ndarray, np.ndarray]: Resampled (planet_data, s1_data)
            """
            if self.resample_strategy == ResampleStrategy.UPSAMPLE_S1:
                  target_shape = planet_data.shape[1:]
                  s1_resampled = []
                  for band in range(s1_data.shape[0]):
                        scaled = cv2.resize(s1_data[band], 
                                          target_shape,
                                          interpolation=cv2.INTER_CUBIC)
                        s1_resampled.append(scaled)
                  return planet_data, np.stack(s1_resampled)
                  
            elif self.resample_strategy == ResampleStrategy.DOWNSAMPLE_PLANET:
                  target_shape = s1_data.shape[1:]
                  planet_resampled = []
                  for band in range(planet_data.shape[0]):
                        scaled = cv2.resize(planet_data[band],
                                          target_shape,
                                          interpolation=cv2.INTER_AREA)
                        planet_resampled.append(scaled)
                  return np.stack(planet_resampled), s1_data

      @staticmethod
      def dynamic_pad(image: torch.Tensor, multiple: int = 32) -> torch.Tensor:
            """
            Pad image to be divisible by the specified multiple.
            
            Args:
                  image (torch.Tensor): Input image tensor
                  multiple (int, optional): Padding multiple. Defaults to 32.
                  
            Returns:
                  torch.Tensor: Padded image tensor
            """
            
            if len(image.shape) == 3:
                  _, h, w = image.shape
            else:
                  h, w = image.shape
            pad_h = (multiple - h % multiple) % multiple
            pad_w = (multiple - w % multiple) % multiple
            
            return torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), 
                                          mode='constant', value=0)

      def handle_nan_values(self, img):
            """
            Replace NaN values in images with the median value.
            
            Args:
                  img: Input image array
                  
            Returns:
                  Processed image array with NaN values replaced
            """
            img = np.nan_to_num(img, nan=np.nanmedian(img))
            return img
      
      def __len__(self) -> int:
            return len(self.file_pairs)

      def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            planet_file, s1_file, gt_file = self.file_pairs[idx]
            
            with rasterio.open(os.path.join(self.planet_dir, planet_file)) as src:
                  planet_data = src.read().astype(np.float32)
                  
            with rasterio.open(os.path.join(self.planet_dir, s1_file)) as src:
                  s1_data = src.read().astype(np.float32)
                  
            with rasterio.open(os.path.join(self.planet_dir, gt_file)) as src:
                  gt_data = src.read(1).astype(np.float32)
            
            # Handle NaN values in both input sources
            planet_data = self.handle_nan_values(planet_data)
            s1_data = self.handle_nan_values(s1_data)
            
            # resample according to strategy
            planet_data, s1_data = self._resample_data(planet_data, s1_data)
            
            # resample to match if needed
            if self.resample_strategy == ResampleStrategy.DOWNSAMPLE_PLANET:
                  gt_data = cv2.resize(gt_data, 
                                    s1_data.shape[1:],
                                    interpolation=cv2.INTER_NEAREST)
            
            # add PlanetScope derived bands
            green = planet_data[1]
            red = planet_data[2]
            nir = planet_data[3]
            
            # Calculate indices
            ndvi = (nir - red) / (nir + red + 1e-10)
            ndwi = (green - nir) / (green + nir + 1e-10)
            savi = (nir - red) * 1.5 / (nir + red + 0.5 + 1e-10)
            
            planet_data = np.vstack([planet_data, ndvi[np.newaxis, ...], 
                                         ndwi[np.newaxis, ...], 
                                         savi[np.newaxis, ...]])
            
            # add Sentinel-1 derived band
            vv = s1_data[0]
            vh = s1_data[1]
            vv_vh_ratio = vv - vh 
            s1_data = np.vstack([s1_data, vv_vh_ratio[np.newaxis, ...]])
            
            # convert to tensors
            planet_tensor = torch.from_numpy(planet_data)
            s1_tensor = torch.from_numpy(s1_data)
            gt_tensor = torch.from_numpy(gt_data).long()
            
            if self.transforms:
                  # use same random seed for consistent transforms
                  seed = np.random.randint(2147483647)
                  torch.manual_seed(seed)
                  planet_tensor = self.transforms(planet_tensor)
                  torch.manual_seed(seed)
                  s1_tensor = self.transforms(s1_tensor)
                  
                  gt_tensor = gt_tensor.unsqueeze(0)
                  torch.manual_seed(seed)
                  gt_tensor = self.transforms(gt_tensor)
                  gt_tensor = gt_tensor.squeeze(0)
                  
            if self.pad:
                  planet_tensor = self.dynamic_pad(planet_tensor)
                  s1_tensor = self.dynamic_pad(s1_tensor)
                  gt_tensor = self.dynamic_pad(gt_tensor)
                  
            return planet_tensor, s1_tensor, gt_tensor
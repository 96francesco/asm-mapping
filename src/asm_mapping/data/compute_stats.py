import os
import numpy as np
import rasterio
import argparse
from typing import Dict, List

BAND_CONFIGS = {
      'planetscope': {
            'bands': {0: 'blue', 1: 'green', 2: 'red', 3: 'nir'},
            'output_file': 'ps_normalization.yaml'
      },
      'sentinel1': {
            'bands': {0: 'vv', 1: 'vh', 2: 'ratio'},
            'output_file': 's1_normalization.yaml'
      }
}

def compute_training_stats(split_dir: str, sensor: str) -> Dict[str, tuple[float, float]]:
      """
      Compute percentiles from training images of a split for each band.
      """
      training_dir = os.path.join(split_dir, "training_set")
      img_folder = os.path.join(training_dir, "images")
      
      # get band configuration for the sensor
      bands = BAND_CONFIGS[sensor]['bands']
      band_values: Dict[int, List] = {band: [] for band in bands.keys()}
      
      for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder, img_name)
            with rasterio.open(img_path, "r") as ds:
                  img = ds.read().astype(np.float32)
                  img = np.nan_to_num(img, nan=np.nanmedian(img))
                  
                  # split values by band
                  for band in bands.keys():
                        if band < 2: 
                              band_values[band].extend(img[band].flatten())
                        elif sensor == 'sentinel1' and band == 2:  # Ratio for Sentinel-1
                              ratio = img[0] - img[1]  # VV - VH in dB scale
                              band_values[band].extend(ratio.flatten())
      
      # compute stats per band
      stats = {}
      for band_idx, band_name in bands.items():
            values = np.array(band_values[band_idx])
            p2 = np.percentile(values, 2)
            p98 = np.percentile(values, 98)
            stats[band_name] = (p2, p98)
      
      return stats

def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--data_dir", type=str, required=True, 
                        help="Base directory containing all splits")
      parser.add_argument("--sensor", type=str, choices=['planetscope', 'sentinel1'],
                        required=True, help="Sensor type")
      args = parser.parse_args()
      
      print(f"\nComputing {args.sensor} normalization values...")
      print("splits_normalization:")
      
      # process each split
      for split_name in sorted(os.listdir(args.data_dir)):
            if split_name.startswith("split_"):
                  split_path = os.path.join(args.data_dir, split_name)
                  print(f"  {split_name}:")
                  stats = compute_training_stats(split_path, args.sensor)
                  for band_name, (p2, p98) in stats.items():
                        print(f"    {band_name}:")
                        print(f"      p2: {p2:.6f}")
                        print(f"      p98: {p98:.6f}")

if __name__ == "__main__":
    main()
import logging
import argparse
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import os

from asm_mapping.train_test_predict.utils import load_config
from asm_mapping.train_test_predict.train_test_split import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--config", required=True, help="Path to config file")
      args = parser.parse_args()

      config = load_config(args.config)
      pl.seed_everything(config["seed"], workers=True)
      config["base_data_dir"] = Path(config["base_data_dir"])
      all_results = []

      for split_n in range(config["n_splits"]):
            logger.info(f"Processing split {split_n}")
            results = train_test_split(config, split_n)
            if results:
                  all_results.append(results)

      metrics_df = pd.DataFrame(all_results)
      mean_metrics = metrics_df.mean()
      std_metrics = metrics_df.std()

      results_path = Path(config["log_dir"]) / f"{config['experiment_name']}_results.csv"
      metrics_df.to_csv(results_path)

      logger.info("\nFinal Results:")
      for metric in mean_metrics.index:
            logger.info(f"{metric}: {mean_metrics[metric]:.3f} ± {std_metrics[metric]:.3f}")
            
      output_path = os.path.join(config['log_dir'], f"{config['experiment_name']}_results.md")
      with open(output_path, 'w') as f:
            f.write(f"# Training Results - {config['experiment_name']}\n\n")
            
            f.write("## Results by Split\n\n")
            for idx, split_results in metrics_df.iterrows():
                  f.write(f"### Split {idx}\n\n")
                  for metric in split_results.index:
                        f.write(f"{metric}: {split_results[metric]:.3f}\n")
                  f.write("\n")
            
            f.write("## Final Results\n\n")
            for metric in mean_metrics.index:
                  f.write(f"{metric}: {mean_metrics[metric]:.3f} ± {std_metrics[metric]:.3f}\n")
                  
if __name__ == "__main__":
      main()

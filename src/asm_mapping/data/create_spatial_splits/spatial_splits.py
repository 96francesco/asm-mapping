import argparse
import logging
from typing import Dict, Any
import random

from asm_mapping.train_test_predict.utils import load_config
from asm_mapping.data.create_spatial_splits.reader import read_tile_info
from asm_mapping.data.create_spatial_splits.blocking import (
    assign_tiles_to_blocks,
    generate_splits,
    validate_splits,
)
from asm_mapping.data.create_spatial_splits.utils import organize_split_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spatial splits for ASM dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input_dir", type=str, help="Input data directory (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--n_splits", type=int, help="Number of splits (overrides config)")
    parser.add_argument("--train_ratio", type=float, help="Training set ratio (overrides config)")
    parser.add_argument("--grid_size", type=int, help="Size of spatial grid (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")

    return parser.parse_args()


def update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update config with command line arguments if provided
    """
    if args.input_dir:
        config["data"]["input_dir"] = args.input_dir
    if args.output_dir:
        config["data"]["output_dir"] = args.output_dir
    if args.n_splits:
        config["splits"]["n_splits"] = args.n_splits
    if args.train_ratio:
        config["splits"]["train_ratio"] = args.train_ratio
    if args.grid_size:
        config["splits"]["grid_size"] = args.grid_size
    if args.seed:
        config["seed"] = args.seed

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required config parameters
    """
    if not config["data"]["input_dir"]:
        raise ValueError("Input directory must be specified in config or command line")
    if not config["data"]["output_dir"]:
        raise ValueError("Output directory must be specified in config or command line")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = update_config(config, args)
    validate_config(config)

    random.seed(config["seed"])

    # read tile information
    logger.info("Reading tile information...")
    tile_info = read_tile_info(config["data"]["input_dir"])

    # create spatial blocks
    logger.info("Creating spatial blocks...")
    tile_to_block = assign_tiles_to_blocks(tile_info, grid_size=config["splits"]["grid_size"])

    # generate splits
    logger.info("Generating splits...")
    splits = generate_splits(
        tile_info,
        tile_to_block,
        n_splits=config["splits"]["n_splits"],
        train_ratio=config["splits"]["train_ratio"],
        random_state=config["seed"],
    )

    # validate the generated splits
    logger.info("Validating splits...")
    if not validate_splits(splits):
        raise ValueError("Invalid splits generated")

    # organize files for each split
    logger.info("Organizing files...")
    for split_idx, split_info in enumerate(splits):
        organize_split_files(tile_info, split_info, config["data"]["output_dir"], split_idx)

    logger.info("Split generation complete!")


if __name__ == "__main__":
    main()

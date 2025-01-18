import shutil
from pathlib import Path
import json
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_split_directories(output_dir: str, split_number: int) -> Dict[str, Path]:
    """
    Create directory structure for a split

    Structure:
    output_dir/
        split_{n}/
            training_set/
                images/
                masks/
            testing_set/
                images/
                masks/
    """
    output_path = Path(output_dir) / f"split_{split_number}"

    # create directories
    directories = {}
    for set_type in ["training_set", "testing_set"]:
        for subdir in ["images", "masks"]:
            dir_path = output_path / set_type / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            directories[f"{set_type}_{subdir}"] = dir_path

    return directories


def save_split_info(split_info: Dict[str, List[str]], output_dir: str, split_number: int) -> None:
    """
    Save split configuration to JSON
    """
    output_path = Path(output_dir) / f"split_{split_number}" / "split_info.json"
    with open(output_path, "w") as f:
        json.dump(split_info, f, indent=2)


def organize_split_files(
    tile_info: Dict[str, Dict],
    split_info: Dict[str, List[str]],
    output_dir: str,
    split_number: int,
) -> None:
    """
    Organize files into training and testing directories
    """
    # create directories
    directories = create_split_directories(output_dir, split_number)

    # copy files to appropriate locations
    for set_type, tile_ids in split_info.items():
        dest_prefix = "training_set" if set_type == "train" else "testing_set"

        for tile_id in tile_ids:
            tile_data = tile_info[tile_id]

            # copy image
            src_img = Path(tile_data["image_path"])
            dest_img = directories[f"{dest_prefix}_images"] / src_img.name
            shutil.copy2(src_img, dest_img)

            # copy mask
            src_mask = Path(tile_data["mask_path"])
            dest_mask = directories[f"{dest_prefix}_masks"] / src_mask.name
            shutil.copy2(src_mask, dest_mask)

    # save split information
    save_split_info(split_info, output_dir, split_number)

    logger.info(f"Organized files for split {split_number}")


def validate_directory_structure(path: Path, required_structure: List[str]) -> bool:
    """
    Validate that a directory contains all required subdirectories
    """
    return all((path / subdir).exists() for subdir in required_structure)

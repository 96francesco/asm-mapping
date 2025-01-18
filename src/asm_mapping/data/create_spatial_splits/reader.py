from pathlib import Path
import rasterio
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_tile_info(data_dir: str) -> Dict[str, Dict[str, Tuple[float, float, float, float]]]:
    """
    Read metadata for all tiles in the dataset

    Args:
        data_dir: Root directory containing 'images' and 'masks' folders

    Returns:
        Dictionary with tile info where:
            - key: tile identifier
            - value: dict with paths and bounds
            {
                'tile_001': {
                    'image_path': 'path/to/image',
                    'mask_path': 'path/to/mask',
                    'bounds': (minx, miny, maxx, maxy)
                }
            }
    """
    data = Path(data_dir)
    images_dir = data / "images"
    masks_dir = data / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError(f"Required directory structure not found in {data_dir}")

    tile_info = {}

    # list all image files
    image_files = sorted(images_dir.glob("*"))

    for img_path in image_files:
        # get tile id (full name) and number
        tile_id = img_path.stem
        tile_number = tile_id.split("_")[1]

        # build mask path with mask_XXX pattern
        mask_path = masks_dir / f"mask_{tile_number}.tif"

        # check if corresponding mask exists
        if not mask_path.exists():
            logger.warning(f"No matching mask found for {tile_id}")
            continue

        # read image bounds
        try:
            with rasterio.open(img_path) as src:
                bounds = src.bounds

            tile_info[tile_id] = {
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "bounds": bounds,
            }

        except Exception as e:
            logger.error(f"Error reading metadata for {tile_id}: {e}")
            continue

    logger.info(f"Successfully read metadata for {len(tile_info)} tiles")
    return tile_info

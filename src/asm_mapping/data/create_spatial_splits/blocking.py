from typing import Dict, List, Tuple
import numpy as np
import logging
# from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data_extent(tile_info: Dict[str, Dict[str, Tuple]]) -> Tuple[float, float, float, float]:
    """
    Get the overall extent of all tiles

    Returns:
        Tuple of (minx, miny, maxx, maxy)
    """
    # Initialize with first tile's bounds
    first_tile = next(iter(tile_info.values()))
    minx, miny, maxx, maxy = first_tile["bounds"]

    # Update with remaining tiles
    for info in tile_info.values():
        bounds = info["bounds"]
        minx = min(minx, bounds[0])
        miny = min(miny, bounds[1])
        maxx = max(maxx, bounds[2])
        maxy = max(maxy, bounds[3])

    return (minx, miny, maxx, maxy)


def assign_tiles_to_blocks(
    tile_info: Dict[str, Dict[str, Tuple]], grid_size: int = 3
) -> Dict[str, int]:
    """
    Create spatial blocks and assign tiles to them based on their location.
    Blocks are created using a regular grid over the data extent.

    Args:
        tile_info: Dictionary containing tile information with bounds
        grid_size: Size of the grid (grid_size x grid_size)

    Returns:
        Dictionary mapping tile_id to block_id
    """
    # get overall extent
    minx, miny, maxx, maxy = get_data_extent(tile_info)

    # calculate grid size based on extent
    x_range = maxx - minx
    y_range = maxy - miny

    # create grid based on config parameter
    n_cells = grid_size
    x_step = x_range / n_cells
    y_step = y_range / n_cells

    # assign tiles to grid cells
    tile_to_block = {}

    # count tiles per block for logging
    block_counts = {i: 0 for i in range(n_cells * n_cells)}

    for tile_id, info in tile_info.items():
        # get tile centroid
        bounds = info["bounds"]
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2

        # calculate grid cell
        x_idx = int((center_x - minx) / x_step)
        y_idx = int((center_y - miny) / y_step)

        # handle edge cases
        x_idx = min(x_idx, n_cells - 1)
        y_idx = min(y_idx, n_cells - 1)

        # assign block ID (flattened grid index)
        block_id = y_idx * n_cells + x_idx
        tile_to_block[tile_id] = block_id
        block_counts[block_id] += 1

    # log distribution of tiles across blocks
    logger.info(f"Distribution of tiles across {n_cells}x{n_cells} grid:")
    for block_id, count in block_counts.items():
        if count > 0:
            logger.info(f"Block {block_id}: {count} tiles")

    return tile_to_block


def generate_splits(
    tile_info: Dict[str, Dict[str, Tuple]],
    tile_to_block: Dict[str, int],
    n_splits: int = 5,
    train_ratio: float = 0.7,
    random_state: int = 42,
    ratio_tolerance: float = 0.02,
) -> List[Dict[str, List[str]]]:
    """
    Generate multiple train-test splits respecting spatial blocks and desired ratio
    """
    # count tiles in each block
    block_sizes = {}
    for block_id in set(tile_to_block.values()):
        block_sizes[block_id] = sum(
            1 for tile_block in tile_to_block.values() if tile_block == block_id
        )

    # calculate target number of training tiles
    total_tiles = len(tile_info)
    np.random.seed(random_state)
    blocks = list(block_sizes.keys())

    splits = []
    max_attempts = 50  # max attempts to find a good split

    for split in range(n_splits):
        best_ratio_diff = float("inf")
        best_split: Dict[str, List[str]] = {"train": [], "test": []}

        # try multiple times to get a good split
        for _ in range(max_attempts):
            np.random.shuffle(blocks)
            current_train_blocks = set()
            current_train_tiles = 0

            # try to get close to target ratio
            for block in blocks:
                # add block if it gets us closer to target
                new_train_tiles = current_train_tiles + block_sizes[block]
                if abs(new_train_tiles / total_tiles - train_ratio) < abs(
                    current_train_tiles / total_tiles - train_ratio
                ):
                    current_train_blocks.add(block)
                    current_train_tiles = new_train_tiles

            # calculate actual ratio
            actual_ratio = current_train_tiles / total_tiles
            ratio_diff = abs(actual_ratio - train_ratio)

            # update best split if this is better
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                test_blocks = set(blocks) - current_train_blocks
                train_tiles = [
                    tile_id
                    for tile_id, block_id in tile_to_block.items()
                    if block_id in current_train_blocks
                ]
                test_tiles = [
                    tile_id
                    for tile_id, block_id in tile_to_block.items()
                    if block_id in test_blocks
                ]
                best_split = {"train": train_tiles, "test": test_tiles}

            if ratio_diff <= ratio_tolerance:
                break

        splits.append(best_split)

        # log split information
        n_train = len(best_split["train"])
        n_test = len(best_split["test"])
        actual_ratio = n_train / (n_train + n_test)
        logger.info(
            f"Split {split + 1}: {n_train} train tiles ({actual_ratio:.2%}), {n_test} test tiles"
        )

    return splits


def validate_splits(splits: List[Dict[str, List[str]]]) -> bool:
    """
    Validate splits have no overlapping tiles and approximate desired ratio
    """
    for i, split in enumerate(splits):
        train_set = set(split["train"])
        test_set = set(split["test"])

        if train_set & test_set:
            logger.error(f"Split {i} has overlapping train/test tiles")
            return False

        total = len(train_set) + len(test_set)
        train_ratio = len(train_set) / total
        logger.info(f"Split {i} train ratio: {train_ratio:.2f}")

    return True

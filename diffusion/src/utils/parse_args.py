"""
Command line argument parsing utility

Parses arguments not supported by fire (e.g., space-separated basin ID lists)
"""
import sys


def parse_masked_basin_ids():
    """
    Parse --masked_basin_ids argument from sys.argv

    Usage example:
        python NsDiff_CAMELS.py --device=cuda --masked_basin_ids 01022500 02069700 11141280 runs --seeds='[1]'

    Returns:
        list[str] | None: list of basin IDs, or None if not specified
    """
    if '--masked_basin_ids' not in sys.argv:
        return None

    idx = sys.argv.index('--masked_basin_ids')
    basin_ids = []

    # Collect all arguments after --masked_basin_ids until the next option or command
    for arg in sys.argv[idx + 1:]:
        if arg.startswith('--') or arg in ('runs', 'show', 'test'):
            break
        basin_ids.append(arg)

    # Remove these arguments from sys.argv so fire won't see them
    sys.argv = sys.argv[:idx] + sys.argv[idx + 1 + len(basin_ids):]

    if basin_ids:
        print(f"Parsed masked_basin_ids: {basin_ids}")

    return basin_ids if basin_ids else None

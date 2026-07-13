"""
Boilerplate code for script building
"""
# Imports
from pathlib import Path
from os import environ
from dotenv import load_dotenv

# Functions
def get_envars(root: Path, fn: str) -> None:
    pass

def parse_path(path: str, relative_to_envar: str | None = None) -> Path:
    """
    Utility function used to derive a path variable in conjunction with
    paths stored as environment variables.

    Parameters
    ----------
    path: str.
        Required. Path string.

    ralative_to_envar: str or None. Default: None.
        Optional. Environment variable to use as the root that the
        `path` argument is considered relative to. If no argument is
        passed, then no environment variable will be called.
    
    Return
    ------
    Path.
    """
    # Convert path argument to a Path instance
    path: Path = Path(path)
    if path.is_absolute() or (relative_to_envar is None):
        # Directly return absolute path
        return path
    # Get relative to path component
    try:
        root = environ[relative_to_envar]
    except KeyError as _:
        msg =\
            "relative_to_envar argument not recognised as an environment "\
            f"variable. Argument passed: {relative_to_envar}."
        raise ValueError(msg) from None
    return Path(root).joinpath(path)
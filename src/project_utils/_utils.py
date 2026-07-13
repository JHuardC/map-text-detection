"""
Boilerplate code for script building
"""
# Imports
from pathlib import Path
from os import environ
from dotenv import load_dotenv
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Functions
def get_envars(root_name: str, root: Path, fn: str = ".env") -> None:
    """
    Load project specific environment variables. Adds project root
    filepath to environment variables.

    Parameters
    ----------
    root_name: str.
        Required. Environment name to place the project root directory
        under.

    root: Path.
        Required. Path to project root directory.
    
    fn: str.
        Optional. File name for the file containing environment
        variables. If no filename is provided, will attempt to load a
        filename of `.env`.
    
    Returns
    -------
    None.
    """
    load_dotenv(root.absolute().joinpath(".env"))
    environ[root_name] = str(root.absolute())


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


def build_argument_parser(filename: str, docstr: str) -> ArgumentParser:
    """
    Builds Command Line Argument Parser with default arguments set.

    Parameters
    ----------
    filename: str.
        Required. Name of the script the function is being called from.

    docstr: str.
        Required. Doc string of script to build argument parser for.
    
    Returns
    -------
    ArgumentParser. Containing optional arguments:
    
    - "-c"/"--config": Optional argument to specify path to a custom
    config json.
    - "-s"/"--stream-level": Optional argument to set level for logging
    messages to be streamed.
    - "-f"/"--file-logs": Boolean flag, if called, will write logging
    messages to file: 'logs/{filename}_YYYYmmDDHHMMSS.log', relative to
    project root dir.
    """
    parser = ArgumentParser(
        description = __doc__, formatter_class = RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-c", "--config",
        action = "store",
        type = str,
        dest = "config",
        metavar = "path/to/config/json",
        default = None,
        help =\
            "Optional. Specify path to config json, containing presets used "\
            "to split tiffs into pngs. Can provide either a relative or "\
            "absolute path; relative paths will be set relative to the "\
            "project root directory. If no argument is provided, will "\
            f"attempt to load config from 'config/{filename}.json', " \
            "relative to project root folder."
    )
    parser.add_argument(
        "-s", "--stream-level",
        action = "store",
        choices = [10, 20, 30, 40, 50],
        default = 20,
        dest = "stream_level",
        help = \
            "Optional. Level for logging messages to be streamed out. "\
            "Default is 20 - info level and above."
    )
    parser.add_argument(
        "-f", "--file-logs",
        action = "store_true",
        dest = "file",
        help = \
            "Optional. Save logging messages to .log file. If flagged, logs "\
            f"will be saved out to 'logs/{filename}_YYYYmmDDHHMMSS.log' "\
            "relative to project root folder. All logging messages will be "\
            "saved (from debug up)."
    )
    return parser

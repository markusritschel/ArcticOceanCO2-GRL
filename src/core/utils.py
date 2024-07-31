# !/usr/bin/env python3
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-07-23
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import functools
import inspect
import logging
import os
from pathlib import Path
import subprocess
import sys
import pytest

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

__all__ = ["setup_logger", "save"]


def setup_logger(level=None, logfile=True, name="root"):
    """Define a logger setup with
    
    - 1x fileHandler: writing log files to :obj:`~src.LOG_DIR` (logfile can be boolean or a file path)
    - 1x streamHandler: streaming logs to terminal

    Parameters
    ----------
    level : int | str
        The log level (according to the logging convention).
    logfile : bool | str
        If True, the log file will be placed in :obj:`~src.LOG_DIR` and named after the calling script (default)
        If logfile is a string, it will be interpreted as a file path (the parent directory must exist)
    
    Return
    ------
    logger
        The logger instance
    """
    from src import LOG_DIR

    caller_file = inspect.stack()[-1].filename
    caller_filename = Path(caller_file).stem

    if not level:
        level = os.getenv('LOGLEVEL', 'INFO').upper()
    print("LOGLEVEL:", level)

    # set up new logger and set level to DEBUG to ensure that all messages are written to the log file
    logger = logging.getLogger()
    logger.setLevel('DEBUG')

    formatter = logging.Formatter('%(asctime)s: '
                                  '[%(levelname)s] '
                                  # '(%(name)s): '
                                  '(%(name)s:#%(lineno)d): '
                                  '%(message)s'
                                  , datefmt='%Y-%m-%d %H:%M:%S')
    
    if logfile:
        if isinstance(logfile, bool):
            logfile = LOG_DIR / f'{caller_filename}_{os.getpid()}.log'
        elif isinstance(logfile, str):
            logdir = Path(logfile).parents[0]
            if not logdir.exists():
                raise IOError(f'Log directory {logdir} does not exist.')
        logger.logfile = logfile.as_posix()
        print(f"Log file: {logger.logfile}")
        
        filehandler = logging.FileHandler(logfile)
        filehandler.setFormatter(formatter)
        filehandler.setLevel('DEBUG')
        logger.addHandler(filehandler)
    else:
        logger.logfile = None

    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)

    logger.info("="*(17+len(caller_file)))
    logger.info("Calling routine: %s", caller_file)
    logger.info("-"*(17+len(caller_file))+"\n")

    return logger


def add_metadata(func):
    """
    A decorator that adds metadata to the function's output.

    The metadata includes the relative path of the file, line number, and git commit hash.

    Parameters
    ----------
    func : callable 
        The function to be decorated.

    Returns
    -------
    callable
        The decorated function.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.setdefault('add_hash', False)

        meta = collect_metadata()
        kwargs['metadata'] = meta

        args = list(args)
        obj = args[0]
        path = Path(args[1])
        suffix = ''
        if kwargs.pop('add_hash'):
            suffix += f"_{meta.git_commit}"
        output_path = f'{path.parent}/{path.stem}{suffix}{path.suffix}'
        args[1] = output_path
        
        obj_type = get_obj_type_str(obj)
        log.info(f"Saved {obj_type} to {output_path}, produced by {meta.relative_code_path}#{meta.line_number} @git-commit:{meta.git_commit}")

        return func(*args, **kwargs)

    def collect_metadata():
        frame = sys._getframe(1).f_back
        code_filename = frame.f_code.co_filename
        line_number = frame.f_lineno #- 1   # TODO: <-- check!
        relative_code_path = os.path.relpath(code_filename)
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

        metadata = {}
        metadata['relative_code_path'] = relative_code_path
        metadata['line_number'] = str(line_number)
        metadata['git_commit'] = git_commit
        return BunchDict(metadata)
        
    return wrapper


class BunchDict(dict):
    """BunchDict is a subclass of the built-in dict class that allows 
    accessing dictionary keys as attributes.

    This class overrides the `__getattr__` and `__setattr__` methods to 
    provide attribute-style access to dictionary keys.
    When an attribute is accessed, it is treated as a dictionary key 
    and the corresponding value is returned.
    When an attribute is set, it is treated as a dictionary key and 
    the corresponding value is updated.

    .. note::
        This is now also implemented in :class:`sklearn.utils.Bunch`

    Example
    -------
    >>> bd = BunchDict()
    >>> bd['key'] = 'value'
    >>> print(bd.key)
    value
    >>> bd.key = 'new value'
    >>> print(bd['key'])
    new value
    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def get_obj_type_str(obj):
    """Transform the output of `type` to a simplified descriptor:
    
    Turns
        "<class 'xarray.core.dataset.Dataset'>"
    into "Dataset"
    """
    return str(type(obj)).split("'")[1].split('.')[-1]


@add_metadata
@functools.singledispatch
def save(obj, *args, **kwargs):
    """
    Save the given object. A message will be printed and written to a log file, including the kwargs that were passed to the save method.

    Parameters
    ----------
    obj : object
        The object to be saved.

    Raises
    ------
    NotImplementedError
        If the according function is not dispatched.

    Notes
    -----
    This function raises a :class:`NotImplementedError` because it is meant to be overridden by subclasses.
    To save objects of a specific type, please use the native method provided by that type.

    Examples
    --------
    >>> pytest.skip()
    >>> save(my_object)
    NotImplementedError: Cannot save object of type <class 'type'> using `save` method. Please use the native method.

    >>> fig = plt.figure()
    >>> ...
    >>> save(fig, "/tmp/myfigure.png", dpi=175)
    Save Figure to /tmp/myfigure.png with dpi=175
    """
    raise NotImplementedError(f"Cannot save object of type {type(obj)} using `save` method. Please use the native method.")


@save.register(plt.Figure)
def _(fig, path, *args, **kwargs):
    plt.savefig(path, *args, **kwargs)

@save.register(pd.DataFrame)
def _(df, path, *args, **kwargs):
    df.to_csv(path, *args, **kwargs)

@save.register(xr.Dataset)
def _(ds, path, *args, **kwargs):
    ds.to_netcdf(path, *args, **kwargs)

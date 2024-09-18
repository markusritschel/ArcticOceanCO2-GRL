# !/usr/bin/env python3
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-07-23
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from .core.utils import save, setup_logger

__version__ = '0.1.0'

# Make some of the basic directories globally available in your environment
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
LOG_DIR  = BASE_DIR / 'logs'
PLOT_DIR = BASE_DIR / 'reports/figures'
jupyter_startup_script = BASE_DIR / 'notebooks/jupyter_startup.ipy'

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

sys.path.append(str(BASE_DIR/"scripts"))


def setup_mpl(latex_columnwidth):
    r"""
    Set up the matplotlib styles and global variables for figure width.
    
    This function reads the style directory located at ``BASE_DIR/"assets/mpl_styles"``.
    It updates the matplotlib style library with the new stylesheets.
    You can check the available styles with :obj:`matplotlib.style.available`.

    The function also sets the global variables ``latex_width``, ``latex_figwidth``, 
    and ``figwidth`` to the calculated figure width based on the ``latex_columnwidth`` parameter.

    Parameters
    ----------
    latex_columnwidth : float
        The width of the column in your :math:`\LaTeX` document. Can be obtained from the :math:`\LaTeX` document
        with ``\showthe\columnwidth``. Look for ``\showthe\columnwidth`` in the log file, copy the output value,
        and insert it here in the function call.

    Example
    -------
    >>> setup_mpl(345.0)
    """
    import matplotlib.pyplot as plt
    new_stylesheets = plt.style.core.read_style_directory(BASE_DIR/"assets/mpl_styles")
    plt.style.core.update_nested_dict(plt.style.library, new_stylesheets)
    plt.style.core.available[:] = sorted(plt.style.library.keys())

    global latex_width, latex_figwidth, figwidth
    latex_width = latex_figwidth = figwidth = latex_columnwidth / 72.27

setup_mpl(latex_columnwidth=397.48499)


def read_config(cfg_file):
    import yaml
    import toml
    from .core.utils import BunchDict
    sfx = Path(cfg_file).suffix
    engine = {'.yaml': yaml.safe_load, 
              '.toml': toml.load}
    with open(cfg_file, 'r') as file:
        config = engine[sfx](file)
    return BunchDict(config)
    

config = read_config(BASE_DIR/'config.toml')


welcome = """
████████╗██╗████████╗██╗     ███████╗
╚══██╔══╝██║╚══██╔══╝██║     ██╔════╝
   ██║   ██║   ██║   ██║     █████╗  
   ██║   ██║   ██║   ██║     ██╔══╝  
   ██║   ██║   ██║   ███████╗███████╗
   ╚═╝   ╚═╝   ╚═╝   ╚══════╝╚══════╝
          Some subtitle ☃
"""
# https://patorjk.com/software/taag/ with "ANSI Shadow" font

# print(pyfiglet.figlet_format("My title", font="slant") + "\n Some subtitle")


if __name__ == '__main__':
    print(welcome)

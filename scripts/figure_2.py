# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-07-23
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
log = logging.getLogger(__name__)


PERIODS = [
    ('1982', '1986'),
    ('1992', '1996'),
    ('2005', '2009'),
    ('2017', '2021')
]

def main():
    ds = xr.open_dataset(DATA_DIR/"raw/merged/observations/pco2/monthly/pco2_MPIM-SOM-FFN_198201-202012.nc")['pco2'].sel(time=slice("1982", "2022"))
    fig_maps, fig_lines = prepare_plot()
    
    plt.show()


def prepare_plot():
    """
    Prepare the plot by creating two figures – one for the map plots and one for the line plots.
    The map figure will have a joint axis in the first row for the colorbar.
    """
    fig_maps = plt.figure(figsize=(12, 3))
    gs = fig_maps.add_gridspec(2, len(PERIODS)+1, height_ratios=[1, 3])
    fig_maps.add_subplot(gs[0, :-1])
    fig_maps.add_subplot(gs[0, -1])
    for i in range(len(PERIODS)+1):
        ax = fig_maps.add_subplot(gs[1, i], projection=ccrs.NorthPolarStereo())
        
    fig_lines, _ = plt.subplots(1, 2, figsize=(12,6), width_ratios=[4,1])
    fig_lines.axes[1].set_axis_off()

    return fig_maps, fig_lines


if __name__ == '__main__':
    setup_logger()
    main()
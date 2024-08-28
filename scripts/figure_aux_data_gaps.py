# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-08-13
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging

from src import DATA_DIR
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import calendar
import matplotlib.pyplot as plt
from my_code_base.plot.maps import *
from src import *

log = logging.getLogger(__name__)

plt.style.use("white_paper")


def main():
    data = read_data()
    data = prepare_data(data)
    plot_maps(data)
    save(plt.gcf(), PLOT_DIR/"aux_data_gaps.png", add_hash=True, transparent=True)


def read_data():
    input_file = (DATA_DIR/"raw/merged/observations/pco2/monthly/").glob(
        "pco2_MPIM-SOM-FFN_*.nc"
    )
    data = xr.open_dataset(next(input_file))
    data = data["pco2"].sel(time=slice("2005", None)).load()
    return data


def prepare_data(data):
    data /= data
    data = data.groupby("time.month").mean("time")
    data = data.sel(lat=slice(50, 90))
    data = -(data.fillna(0) - 1)
    data = data.where(data==1)
    return data


def plot_maps(data):
    g = data.plot.contourf(
        x="lon",
        y="lat",
        col="month",
        col_wrap=3,
        add_colorbar=False,
        levels=[0, 1],
        colors="crimson",
        transform=ccrs.PlateCarree(),
        subplot_kws=dict(projection=ccrs.NorthPolarStereo(),),
        figsize=(figwidth, figwidth*1.2),
    )
    for i, ax in enumerate(g.axs.flat):
        ax.polar.add_features(
            labels=False,
            ruler_kwargs=dict(primary_color='#666', width=1),
            ocean_kwargs=dict(facecolor="#fff"),
            land_kwargs=dict(facecolor="#ddd"),
        )
        ax.set_title(calendar.month_name[i + 1], fontsize='small')


if __name__ == "__main__":
    main()

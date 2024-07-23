# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-07-23
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
"""This script creates Figure 1 of the paper"""
import logging

import cartopy.crs as ccrs
import cmcrameri
import cmocean
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
from my_code_base.plot.maps import *

from src import DATA_DIR, PLOT_DIR
from src.core.utils import save, setup_logger
from src.helper.coordinates import adjust_lons


log = logging.getLogger(__name__)


def main():
    ds = read_socat_obs_data(DATA_DIR/"raw/SOCATv2022_tracks_gridded_monthly.nc")['fco2_count_nobs']
    ds = ds.sel(time=slice("1980", "2021"))
    ds_50N = ds.sel(lat=slice(50,90))
    ds_66N = ds.sel(lat=slice(66,90))
    df_66N_avg = transform_to_df(ds_66N)
    
    fig = prepare_plot()
    ax1, ax2, ax3 = fig.axes
    plot_map(ds_50N, ax=ax1)
    plot_cum_hist(df_66N_avg, ax=ax2)
    plot_heatmap(df_66N_avg, ax=ax3)
    save(fig, PLOT_DIR/f"figure_1.png", dpi=300)


def read_socat_obs_data(filepath):
    socat_ds = xr.open_dataset(filepath, chunks={})
    socat_ds = socat_ds.rename({'tmnth': 'time', 'xlon': 'lon', 'ylat': 'lat'})    
    socat_ds = adjust_lons(socat_ds, 'lon')
    socat_ds['time']= pd.to_datetime(socat_ds.time.dt.strftime('%Y-%m-15').values)
    socat_ds = socat_ds.sel(time=slice("1980", "2022"))
    return socat_ds


def transform_to_df(ds):
    ds = ds.sum(['lon', 'lat']) / 1000
    df = ds.to_dataframe()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df = df.pivot(index='year', columns='month', values='fco2_count_nobs')
    return df


def prepare_plot():
    fig = plt.figure(figsize=(8, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1])
    _ = fig.add_subplot(gs[0, :], projection=ccrs.NorthPolarStereo())
    a = fig.add_subplot(gs[1, 0])
    _ = fig.add_subplot(gs[1, 1], sharex=a)
    return fig


def plot_map(ds, ax):
    ds = ds.sum('time')
    ds.plot(ax=ax, vmax=15_000, transform=ccrs.PlateCarree(), 
            cmap=cmocean.cm.dense, 
            cbar_kwargs={'shrink': .6, 'label': '# observations'})
    ax.polar.add_features()


def plot_cum_hist(df, ax):
    df = df.sum(axis=1)
    ax.step(df.index, df, where='mid', color='k')
    ax.fill_between(df.index, df, step="mid", alpha=0.1, fc='k')
    ax.text(0, .85, "Number of CO2 observations\n[in ×10³]", transform=ax.transAxes, va='top')
    ax.set_xlim(1980, 2021)
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    sns.despine(offset={'left':10}, trim=True, ax=ax)


def plot_heatmap(df, ax):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    vmin = 0
    if (quantile:=np.nanquantile(df.values, 0.98))==0:
        vmax_ = np.nanmax(df.values)
    else:
        vmax_ = quantile

    im = ax.pcolor(df.index, df.columns, df.T, vmin=vmin, vmax=vmax_, cmap=cmcrameri.cm.grayC_r)
    ax.set_yticks(np.arange(1, 13), ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

    axins = inset_axes(ax,
                       width="40%",
                       height="5%",
                       loc='upper left',
                       bbox_to_anchor=(0.0, -0.075, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )
    plt.colorbar(im, cax=axins, label='', extend='max', orientation='horizontal')
    ax.invert_yaxis()
    sns.despine(offset={'left':10}, trim=True, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')


if __name__ == '__main__':
    log = setup_logger()
    main()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-07-23
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
"""This script creates Figure 1 of the paper"""
import logging
import textwrap

import cartopy.crs as ccrs
import cmcrameri
import cmocean
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from my_code_base.plot.maps import *

from src import *
from src.core.utils import save, setup_logger
from src.helper.coordinates import adjust_lons

log = logging.getLogger(__name__)

plt.style.use(BASE_DIR/"assets/mpl_styles/white_paper.mplstyle")


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
    save(fig, PLOT_DIR/"figure_1.png", dpi=300, transparent=True)


def read_socat_obs_data(filepath):
    socat_ds = xr.open_dataset(filepath, chunks={})
    socat_ds = socat_ds.rename({'tmnth': 'time', 'xlon': 'lon', 'ylat': 'lat'})    
    socat_ds = adjust_lons(socat_ds, 'lon')
    socat_ds['time']= pd.to_datetime(socat_ds.time.dt.strftime('%Y-%m-15').values)
    socat_ds = socat_ds.sel(time=slice("1980", "2022"))
    return socat_ds


def transform_to_df(ds):
    """Transform the xarray dataset to a pandas DataFrame.
    The dataset is first summed over the lon and lat dimensions and scaled by a factor of 1000.
    """
    ds = ds.sum(['lon', 'lat']) / 1000
    df = ds.to_dataframe()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df = df.pivot(index='year', columns='month', values='fco2_count_nobs')
    return df


def prepare_plot():
    fig = plt.figure(figsize=(8, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1])
    map_ax = fig.add_subplot(gs[0, :], projection=ccrs.NorthPolarStereo())
    hist_1d_ax = fig.add_subplot(gs[1, 0])
    hist_2d_ax = fig.add_subplot(gs[1, 1], sharex=hist_1d_ax)
    return fig


def plot_map(ds, ax):
    ds = ds.sum('time')
    ds.plot(ax=ax, vmax=15_000, transform=ccrs.PlateCarree(), 
            cmap=cmocean.cm.dense, 
            cbar_kwargs={'shrink': .6, 'label': '# observations'})
    add_region_info(ax)
    ax.polar.add_features(ruler_kwargs={'width': .3, 'primary_color': '#333'})


def add_region_info(ax):
    gdf = gpd.read_file(BASE_DIR/"assets/arctic-regions/arctic-regions.shp").iloc[4:]
    gdf = gdf.set_crs('epsg:4326')  # Assuming the original CRS is WGS84 (EPSG:4326)
    gdf = gdf.to_crs(epsg=3995)  # Example: transform to Arctic Polar Stereographic

    gdf.apply(lambda x: ax.annotate(text='\n'.join(textwrap.wrap(x['name'], width=15)), 
                                    xy=x.geometry.centroid.coords[0], 
                                    ha='center', fontsize='x-small',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='none', alpha=0.85),
                                    zorder=100),
                axis=1)
    gdf.plot(ax=ax, facecolor='none', edgecolor='orange', lw=1.5)


def plot_cum_hist(df, ax):
    df = df.sum(axis=1)
    ax.step(df.index, df, where='mid', color='k')
    ax.fill_between(df.index, df, step="mid", alpha=0.1, fc='k')
    ax.set_title("Number of CO2 observations north of 66°N [in ×10³]", loc='left', y=1.1)
    ax.set_xlim(1980, 2021)
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    sns.despine(offset={'left':10}, trim=True, ax=ax)


def plot_heatmap(df, ax):
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
    plt.colorbar(im, cax=axins, label='', extend='max', extendfrac=.2, orientation='horizontal')
    ax.invert_yaxis()
    sns.despine(offset={'left':10}, trim=True, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')


if __name__ == '__main__':
    log = setup_logger()
    main()

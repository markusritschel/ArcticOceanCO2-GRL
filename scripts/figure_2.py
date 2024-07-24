# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-07-23
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import seaborn as sns
from src.core.utils import setup_logger
from src import *

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
    ax = fig_lines.axes[0]
    plot_field_averages(ds, ax=ax)
    add_period_highlights(ax)
    sns.despine(offset={'left': 5}, trim=True, ax=ax)
    
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


def plot_field_averages(ds, ax):
    domains = {
        'global oceanic pCO2': [-90, 90],
        'NH oceanic pCO2': [0, 90],
        'Arctic Ocean pCO2': [66, 90],
    }
    for domain, lat_range in domains.items():
        ds_domain = ds.sel(lat=slice(*lat_range))
        ds_domain = compute_weighted_mean(ds_domain)
        ds_domain.plot(ax=ax, label=domain)
        ds_ = ds_domain.stats.weighted_mean('time')
        ds_['year'] = pd.to_datetime(ds_.year, format='%Y')
        ds_.plot.step(where='post', ax=ax)
        # Add legend
        # Complete annual mean until end of series
        # Add atmospheric pCO2
        ...
    ax.set_xlabel('')
    ax.set_ylabel('pCO2 [µatm]')
    ax.set_title('')


def add_period_highlights(ax):
    for period in PERIODS:
        ax.axvspan(*period, fc='grey', alpha=.15)
    ax.axvline(pd.to_datetime('2005'), ls='--', lw=1, c='grey')
    ax.text(pd.to_datetime('2004-01-01'), 280, "total data points collected\nuntil 2005: approx. 40,000", ha='right', va='top', fontsize='small')


def compute_weighted_mean(ds, dims=['lon','lat']):
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = 'weights'
    ds_weighted = ds.weighted(weights)
    return ds_weighted.mean(dims)




if __name__ == '__main__':
    setup_logger()
    main()
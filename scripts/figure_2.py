# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-07-23
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging
from functools import cache

import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import xarrayutils
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from my_code_base.plot.maps import *
from my_code_base.stats.xarray_utils import *

from src import *
from src.core.utils import setup_logger
from src.helper.graphics import get_common_norm

log = logging.getLogger(__name__)

plt.style.use(BASE_DIR/"assets/mpl_styles/white_paper.mplstyle")


PERIODS = [
    ('1982', '1986'),
    ('1992', '1996'),
    ('2005', '2009'),
    ('2017', '2021')
]


def main():
    ds = xr.open_dataset(DATA_DIR/"raw/merged/observations/pco2/monthly/pco2_MPIM-SOM-FFN_198201-202012.nc")['pco2'].sel(time=slice("1982", "2022"))
    fig_maps, fig_lines = prepare_plot()
    
    plot_map_panels(ds, fig=fig_maps)

    ax = fig_lines.axes[0]
    plot_field_averages(ds, ax=ax)
    add_period_highlights(ax)
    sns.despine(offset={'left': 5}, trim=True, ax=ax)

    save(fig_maps, PLOT_DIR/'figure_2_maps.png', dpi=300, transparent=True, add_hash=True)
    save(fig_lines, PLOT_DIR/'figure_2_lines.png', dpi=300, transparent=True, add_hash=True)
    

def prepare_plot():
    """
    Prepare the plot by creating two figures – one for the map plots and one for the line plots.
    The map figure will have a joint axis in the first row for the colorbar of the avg maps.
    """
    fig_maps = plt.figure(figsize=(12, 3))
    gs = fig_maps.add_gridspec(2, len(PERIODS)+1, height_ratios=[1, 4], hspace=.25)
    fig_maps.add_subplot(gs[0, :-1])
    fig_maps.add_subplot(gs[0, -1])
    for i in range(len(PERIODS)+1):
        fig_maps.add_subplot(gs[1, i], projection=ccrs.NorthPolarStereo())
        
    fig_lines, _ = plt.subplots(1, 2, figsize=(12,5), width_ratios=[4,1])
    fig_lines.axes[1].set_axis_off()
    # plt.subplots_adjust(right=0.7)

    return fig_maps, fig_lines


def plot_field_averages(ds, ax):
    domains = {
        'Global ocean': [-90, 90],
        'NH ocean': [0, 90],
        'Ocean north of 66°N': [66, 90],
    }
    colors = {
        "Global ocean": "#901886",
        "NH ocean": "#54b72d",
        "Ocean north of 66°N": "#1e6daa",
        "Global atmosphere": "#F09201",
    }
    label_positions = iter([.8, .7, .6, .9])
    for domain, lat_range in domains.items():
        color = colors[domain]

        ds_domain = ds.sel(lat=slice(*lat_range))
        ds_domain = compute_weighted_mean(ds_domain)
        ds_domain.plot(ax=ax, c=color, lw=.8)
        ds_ = ds_domain.stats.weighted_mean('time')
        ds_ = ds_.stats.fill_months_with_annual_value()
        ds_.plot.step(where='post', ax=ax, c=color, label=domain)
        annotate_timeseries(ax, domain, ds_.isel(time=-1).values, next(label_positions), color)
        compute_avg_pCO2(ds_, domain)

    domain = "Global atmosphere"
    color = colors[domain]
    atm_co2 = get_atmospheric_co2().loc["1982":"2020"]
    atm_co2.plot(ax=ax, label=domain, ls='--', c=color, legend=False)
    compute_avg_pCO2(atm_co2.to_xarray()['deseason'], 'atmosphere')
    annotate_timeseries(ax, domain, atm_co2.iloc[-1,0], next(label_positions), color)
    # TODO: Retrieve atmospheric pCO2 data automatically (see websites)

    ax.set_xlabel('')
    ax.set_ylabel('pCO2 [µatm]')
    ax.set_title('')


def compute_avg_pCO2(da, domain):
        avg_05_09 = da.sel(time=slice('2005', '2009')).mean('time').values
        avg_17_21 = da.sel(time=slice('2017', '2021')).mean('time').values
        print('-'*60)
        print(f"avg pCO2 {domain} (2005–2009): {avg_05_09:.2f}")
        print(f"avg pCO2 {domain} (2012–2021): {avg_17_21:.2f}")
        print(f"==> ΔpCO2 {domain}: {avg_17_21 - avg_05_09:.2f}")


def add_period_highlights(ax):
    for period in PERIODS:
        ax.axvspan(*period, fc='grey', alpha=.15)
    ax.axvline(pd.to_datetime('2005'), ls='--', lw=1, c='grey')
    ax.text(pd.to_datetime('2004-01-01'), 275, "total data points collected until 2005: approx. 40,000", ha='right', va='top', fontsize='small')


def compute_weighted_mean(ds, dims=['lon','lat']):
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = 'weights'
    ds_weighted = ds.weighted(weights)
    return ds_weighted.mean(dims)


def plot_map_panels(ds, fig):
    map_axes = fig.axes[2:]
    images=[]
    for period, ax in zip(PERIODS, map_axes[:-1]):
        im = plot_average_map(ds, period, ax)
        images.append(im)
    cbar_avg_maps(fig.axes[0], images)

    im = plot_trend_map(ds, map_axes)
    cbar_trend_map(fig.axes[1], im)


def plot_average_map(ds, period, ax):
    ds_period = ds.sel(time=slice(*period))
    ds_avg = ds_period.mean('time')

    print('-'*60)
    sub_arctic_median = ds_avg.sel(lat=slice(50, 66)).median()
    print(f"Sub-Arctic median: {sub_arctic_median.values} [µatm]")
    
    high_arctic_percentiles = ds_period.sel(lat=slice(66, 90)).quantile([.02, .98], dim='time').mean(['lon','lat']).values
    print(f"High-Arctic percentiles: {high_arctic_percentiles}")
    high_arctic_percentile_range = high_arctic_percentiles[1] - high_arctic_percentiles[0]
    print(f"High-Arctic percentile range (98th - 2nd): {high_arctic_percentile_range:.2f} µatm")
    
    im = ds_avg.plot(ax=ax, transform=ccrs.PlateCarree(), 
                         add_colorbar=False, robust=True)
    ax.polar.add_features(ruler=False, labels=False)
    ax.set_title(f'{period[0]}–{  period[1]}', y=-.2)
    return im


def plot_trend_map(ds, map_axes):
    ds_trend = ds.sel(time=slice("2005", "2021")).resample(time='1YS').mean()
    ds_trend = xarrayutils.linear_trend(ds_trend, dim='time').slope
    ax = map_axes[-1]
    im = ds_trend.plot(ax=ax, transform=ccrs.PlateCarree(), 
                  cmap=cmocean.cm.balance,
                  add_colorbar=False, robust=True)
    ax.polar.add_features(ruler=False, labels=False)
    ax.set_title('2005–2021 trend', y=-.2)
    return im


def cbar_avg_maps(ax, images):
    cbar_ax = inset_axes(ax, width="30%", height="30%", loc="center")
    ax.set_axis_off()
    norm = get_common_norm(images)
    for im in images:
        im.set_norm(norm)
    cbar = plt.colorbar(images[0], cax=cbar_ax, norm=norm,
                 orientation='horizontal', shrink=.4, extend='both', extendfrac=.2
                 )
    cbar.set_label('pCO2 [µatm]', labelpad=10)
    cbar_ax.xaxis.set_ticks_position('bottom')
    cbar_ax.xaxis.set_label_position('top')


def cbar_trend_map(ax, im):
    cbar_ax = inset_axes(
        ax,
        width="100%",
        height="30%",
        loc="center",
    )
    ax.set_axis_off()
    cbar = plt.colorbar(im, cax=cbar_ax,
                 orientation='horizontal', shrink=.4, extend='both', extendfrac=.2
                 )
    cbar.set_label('trend [µatm/year]', labelpad=10)
    cbar_ax.xaxis.set_ticks_position('bottom')
    cbar_ax.xaxis.set_label_position('top')


def annotate_timeseries(ax, label, y, ytext, color):
    ax.annotate(label, 
                xy=('2021', y),
                xycoords='data',    
                xytext=(1.05, ytext),
                textcoords='axes fraction',
                va='center',
                ha='left',
                color=color,
                arrowprops=dict(arrowstyle='-', 
                                lw=.5,
                                ec=color, 
                                relpos=(0, .25),
                                shrinkA=3, 
                                )
                )


@cache
def get_atmospheric_co2():
    #TODO:Implement this with automatic retrieval from other sources 
    from io import StringIO

    from requests import get
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    csv_input = StringIO(get(url).text)  # this prevents an HTTPs forbidden error
    df = pd.read_csv(csv_input, comment='#', sep=r"\s+",
                     names=['year', 'month', 'date_dec', 'monthly_avg', 'deseason', 'days', 'days_stdev', 'unc_mon_mean'],
                     usecols=range(5)).drop(['date_dec'], axis=1)

    # Combine 'year' and 'month' into a single datetime column
    df['time'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df.set_index('time', inplace=True)
    df.drop(['year', 'month'], axis=1, inplace=True)

    return df


if __name__ == '__main__':
    setup_logger(level='INFO')
    main()

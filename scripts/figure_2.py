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
    ('2016', '2020')
]


def main():
    ds = xr.open_dataset(DATA_DIR/"raw/merged/observations/pco2/monthly/pco2_MPIM-SOM-FFN_198201-202012.nc")['pco2']
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
        report_avg_pCO2(ds_, domain)

    domain = "Global atmosphere"
    color = colors[domain]
    _start, _end = ds.time.dt.year[0].values, ds.time.dt.year[-1].values
    atm_co2 = get_atmospheric_co2().loc[str(_start):str(_end)]
    atm_co2.plot(ax=ax, label=domain, ls='--', c=color, legend=False)
    annotate_timeseries(ax, domain, atm_co2.iloc[-1,0], next(label_positions), color)
    report_avg_pCO2(atm_co2.to_xarray()['deseason'], 'atmosphere')
    # TODO: Retrieve atmospheric pCO2 data automatically (see websites)

    ax.set_xlabel('')
    ax.set_ylabel('pCO2 [µatm]')
    ax.set_title('')


def report_avg_pCO2(da, domain):
        start1 = 2005
        end1 = start1 + 5
        avg_1 = da.sel(time=slice(str(start1), str(end1))).mean('time').values
        
        end2 = da.time.dt.year[-1].values
        start2 = end2 - 5
        avg_2 = da.sel(time=slice(str(start2), str(end2))).mean('time').values
        
        print('-'*60)
        print(f"avg pCO2 {domain} ({start1}–{end1}): {avg_1:.2f}")
        print(f"avg pCO2 {domain} ({start2}–{end2}): {avg_2:.2f}")
        print(f"==> ΔpCO2 {domain}: {avg_2 - avg_1:.2f}")


def add_period_highlights(ax):
    for start, end in PERIODS:
        ax.axvspan(start, str(int(end)+1), fc='grey', alpha=.15)

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
    stats_dict = {}
    
    for period, ax in zip(PERIODS, map_axes[:-1]):
        im, stats = plot_average_map(ds, period, ax)
        images.append(im)
        stats_dict.update(stats)
    cbar_avg_maps(fig.axes[0], images)

    im = plot_trend_map(ds, map_axes)
    cbar_trend_map(fig.axes[1], im)

    print(pd.DataFrame.from_dict(stats_dict, orient='index').round(2))


def plot_average_map(ds, period, ax):
    ds_period = ds.sel(time=slice(*period))
    ds_avg = ds_period.mean('time')


    im = ds_avg.plot(ax=ax, transform=ccrs.PlateCarree(), 
                         add_colorbar=False, robust=True)
    ax.polar.add_features(ruler=False, labels=False)
    ax.set_title(f'{period[0]}–{  period[1]}', y=-.2)

    # mask out the Baltic Sea
    # data = ds_avg
    # masks = mask_xarray(data, BASE_DIR/"assets/arctic-regions/arctic-regions.shp")
    # data = data.where(~masks.sel(domain='Hudson Bay'))
    # data = data.where(~masks.sel(domain='Baltic Sea'))

    stats = gen_regional_stats(ds_avg, period)

    return im, stats


def gen_regional_stats(data, period):
    stats_dict = dict()

    sub_arctic_avg = data.sel(lat=slice(50, 66)).median()
    stats_dict["sub_arctic_avg"] = sub_arctic_avg.values

    sub_arctic_percentiles = data.sel(lat=slice(50, 66)).quantile([0.02, 0.98]).values
    stats_dict["sub_arctic_percentile_02"] = sub_arctic_percentiles[0]
    stats_dict["sub_arctic_percentile_98"] = sub_arctic_percentiles[1]
    stats_dict["sub_arctic_percentile_range"] = (
        sub_arctic_percentiles[1] - sub_arctic_percentiles[0]
    )

    high_arctic_avg = data.sel(lat=slice(66, 90)).median()
    stats_dict["high_arctic_avg"] = high_arctic_avg.values

    high_arctic_percentiles = data.sel(lat=slice(66, 90)).quantile([0.02, 0.98]).values
    stats_dict["high_arctic_percentile_02"] = high_arctic_percentiles[0]
    stats_dict["high_arctic_percentile_98"] = high_arctic_percentiles[1]
    stats_dict["high_arctic_percentile_range"] = (
        high_arctic_percentiles[1] - high_arctic_percentiles[0]
    )

    return {f"{period[0]}–{period[1]}": stats_dict}


def plot_trend_map(ds, map_axes):
    ds_trend = ds.sel(time=slice("2005", None)).resample(time='1YS').mean()
    ds_trend = xarrayutils.linear_trend(ds_trend, dim='time').slope
    ax = map_axes[-1]
    im = ds_trend.plot(ax=ax, transform=ccrs.PlateCarree(), 
                  cmap=cmocean.cm.balance,
                  add_colorbar=False)
    ax.polar.add_features(ruler=False, labels=False)
    end_year = ds.time.dt.year[-1].values
    ax.set_title(f'2005–{end_year} trend', y=-.2)
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
                 orientation='horizontal', shrink=.4,
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

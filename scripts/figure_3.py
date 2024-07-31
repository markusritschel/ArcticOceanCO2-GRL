# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-07-30
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import calendar
import logging
import cartopy.crs as ccrs
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import xarrayutils
import xeofs
from matplotlib import pyplot as plt
import matplotlib as mpl
import mpl_axes_aligner

from src import BASE_DIR, DATA_DIR, PLOT_DIR
from src.core.utils import save
from src.helper.datahandling import mask_xarray
from my_code_base.plot.maps import *


log = logging.getLogger(__name__)

plt.style.use(BASE_DIR/"assets/mpl_styles/white_paper.mplstyle")


def main(modes=[1,2]):
    n_modes = len(modes)
    data = read_data()
    data = prepare_data(data)
    eofs, pcs, expvar = perform_eof_analysis(data, n_modes=10)
    fig = prepare_figure(modes)
    fig_eof = fig.subfigs[0]
    fig_pc = fig.subfigs[1]

    images = []
    for i, mode in enumerate(modes):
        eof_ax = fig_eof.axes[i]
        pc_ax = fig_pc.axes[2*i]
        pc_seas_ax = fig_pc.axes[2*i+1]
        
        eof = eofs.sel(mode=mode)
        pc = pcs.sel(mode=mode)
        
        im = plot_eof_map(eof=eof, ax=eof_ax)
        images.append(im)
        plot_pval_hatch(pc, data, ax=eof_ax)
        plot_principal_component(pc=pc, ax=pc_ax)
        plot_pc_seasonality(pc=pc, ax=pc_seas_ax)
        # plot_homogenous_map(ax=hom_map_ax)

    adjust_ticks(fig)
    add_colorbar(images, fig_eof)

    save(fig, PLOT_DIR/"figure_3.png", add_hash=True)


def prepare_figure(modes):
    n_modes = len(modes)
    fig = plt.figure(figsize=(12, 8))#, layout='constrained')
    fig_eof, fig_pc = fig.subfigures(1, 2, width_ratios=[1, 2], hspace=0)

    fig_eof.subplots(n_modes, 1, subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
    fig_pc.subplots(n_modes, 2, width_ratios=[5, 2], height_ratios=[1,]*n_modes, sharey=True)
    fig_pc.subplots_adjust(hspace=.2, wspace=0.05, top=.75, bottom=.25, left=0)
    # fig_eof.set_facecolor('#f7d4d4')
    # fig_pc.set_facecolor('#d4eef7')
    return fig


def read_data():
    input_file = (DATA_DIR/f"raw/merged/observations/pco2/monthly/").glob(f"pco2_MPIM-SOM-FFN_*.nc")
    data = xr.open_dataset(next(input_file))
    data = data['pco2'].sel(time=slice('2005', '2022')).load()
    data['time'] = pd.to_datetime(data.time) + pd.DateOffset(days=-14)
    return data


def prepare_data(data: xr.DataArray):
    """
    Prepare the data for the EOF analysis:

    - Remove linear trend of each grid point. This also centers the data around 0, i.e. it 
    removes the long-term mean.
    - Mask out the Baltic Sea
    - Restrict the domain to the Arctic
    - Fill NaNs with the mean of the detrended time series (--> 0) to remove variance
    """
    data = xarrayutils.xr_detrend(data, dim='time')

    # mask out the Baltic Sea
    masks = mask_xarray(data, BASE_DIR/"assets/arctic-regions/arctic-regions.shp")
    data = data.where(~masks.sel(domain='Baltic Sea'))

    # Restrict the domain to the Arctic
    data = data.sel(lat=slice(50, 90))

    # Fill NaNs with the mean of the detrended time series (--> 0) to remove variance
    data = data.fillna(data.mean('time'))

    return data


def perform_eof_analysis(data, n_modes=10):
    model = xeofs.models.EOF(n_modes=n_modes, standardize=False, center=True, use_coslat=True)
    model.fit(data, dim=['time'])

    rotator = xeofs.models.EOFRotator(n_modes=n_modes, power=1)
    rotator.fit(model)

    expvar = rotator.explained_variance_ratio() * 100

    try:
        n_PC = np.where(expvar.cumsum() >= 70)[0][0] + 1
        print(f"The first {n_PC} modes explain at least 70% of the variance.")
    except:
        log.warning(f"Less than 70% of the variance explained by the first {len(expvar)} EOFs.")

    plot_expvar(expvar)
    
    eofs = rotator.components()
    pcs = rotator.scores()

    return eofs, pcs, expvar


def plot_expvar(expvar):
    fig = plt.figure()
    expvar.plot(marker='.', lw=.5, ax=plt.gca())
    plt.title('')
    save(fig, PLOT_DIR/"eof_analysis_explained_variance.png")


def plot_eof_map(eof, ax):
    im = eof.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmocean.cm.balance, add_colorbar=False)
    ax.set_title('')
    ax.polar.add_features(labels=False, 
                          ruler_kwargs={'primary_color': '#808080', 
                                        'secondary_color': '#EFEFDA'})
    return im


def plot_pval_hatch(pc, data, ax):
    pval = xarrayutils.xr_linregress(pc, data, dim='time').p_value
    pval.where(pval < .05).plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                            add_colorbar=False,
                                            # levels=[.05,1], lw=2, colors='w'
                                            # )
                                            cmap='none',  # alpha=.7,
                                            hatches=['...'])
    ax.set_title('')


def plot_principal_component(pc, ax):
    pc.plot(ax=ax, c='k')
    ax.axhline(0, c='.5', lw=1, zorder=0)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_title('')

def adjust_ticks(fig):
    fig_eof = fig.subfigs[0]
    n_modes = len(fig_eof.axes)
    fig_pc = fig.subfigs[1]

    for ax in fig_pc.axes:
        ax.tick_params(top=True, bottom=True, direction='in')
        dx = 0/72.
        dy = -5/72.
        offset = mpl.transforms.ScaledTranslation(dx, dy, fig_pc.dpi_scale_trans)
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
    
    axes_without_twinx = fig_pc.axes[:-n_modes]
    last_axes_row = axes_without_twinx[-(len(axes_without_twinx)//n_modes):]
    for ax in last_axes_row:
        ax.set_xticklabels([])


def add_colorbar(images, fig):
    for im in images:
        # TODO: Normalize images to the same color scale
        continue
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    # cbar.set_label('Colorbar Label')


def plot_pc_seasonality(pc, ax):
    ax.set_xticks(np.arange(12)+1, [calendar.month_abbr[m][0] for m in np.arange(12)+1])
    ax_sia = plt.twinx(ax)
    seasonal_pc = calc_seasonality(pc.to_pandas())
    get_nyr_averages(seasonal_pc.T, 4).T.plot(ax=ax, cmap='PuRd', legend=False)

    add_sia_to_pc_seasonality(ax_sia)
    mpl_axes_aligner.align.yaxes(ax, 0, ax_sia, 6.5)

    ax.axhline(0, c='.5', lw=1, zorder=0)
    ax.tick_params(top=True, bottom=True, direction='in')

    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')


def add_sia_to_pc_seasonality(ax):
    ice_da = xr.open_dataset("/home/markusritschel/PycharmProjects/corsica/data/processed/monthly/observations_siarea.nc").sel(time=slice("2005", "2021")).load().siarea
    sia_da = ice_da.sel(region='arctic', member='NASA-Team')
    sia_da['time'] = pd.to_datetime(sia_da.time) + pd.DateOffset(days=-14)

    sia_seas_offset = sia_da.mean('time').values
    seasonal_sia = calc_seasonality(sia_da.interpolate_na('time').to_pandas()) + sia_seas_offset
    
    seasonal_sia.mean(axis=1).plot(ax=ax, ls='--', lw=1, c='.5', zorder=0, legend=False)

    ax.set_ylabel("SIA [million km²]")
    ax.annotate("SIA\n[million km²]", xy=(5, seasonal_sia.mean(axis=1)[5]), xytext=(.5,.9),
                textcoords=ax.transAxes,
                ha='left', va='top',
                color='.5',
                fontsize='x-small',
                arrowprops=dict(arrowstyle="-", lw=.5, color='.5'),
                )


def get_nyr_averages(df, n=5):
    """df should have dimensions (rows: years, cols: months)"""
    grouped = df.groupby((df.index - df.index[0])//n)
    df_avg = grouped.mean()
    df_avg.index = [f"{v.index[0]} – {v.index[-1]}" for k, v in grouped]
    return df_avg


def calc_seasonality(df):
    from statsmodels.tsa.stl._stl import STL
    results = STL(df, period=12, robust=True).fit()

    seasonal = results.seasonal.to_frame()
    seasonal['month'] = seasonal.index.month
    seasonal['year'] = seasonal.index.year
    seasonal = seasonal.pivot(columns='year', index='month')

    return seasonal.droplevel(0, axis=1)#.mean(axis=1)


def plot_homogenous_map(ax):
    ...



if __name__ == "__main__":
    main()

# %%

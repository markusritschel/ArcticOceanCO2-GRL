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
    data = read_data()
    data = prepare_data(data)
    eof_model = perform_eof_analysis(data, n_modes=10, plot_expvar=False)

    fig = prepare_figure(modes)
    fig_eof = fig.subfigs[0]
    fig_hmv = fig.subfigs[1]
    fig_pc = fig.subfigs[2]

    eof_images = []
    hmv_images = []
    for i, mode in enumerate(modes):
        eof_ax = fig_eof.axes[i]
        hmv_ax = fig_hmv.axes[i]
        pc_ax = fig_pc.axes[2*i]
        pc_seas_ax = fig_pc.axes[2*i+1]
        
        eof = eof_model.components().sel(mode=mode)
        pc = eof_model.scores().sel(mode=mode)

        im = plot_eof_map(eof, ax=eof_ax)
        eof_images.append(im)
        add_pval_hatch(pc, data, ax=eof_ax)
        title_dict = {1: 'high-Arctic mode EOF', 2: 'sub-Arctic mode EOF'}
        eof_ax.set_title(title_dict[int(eof.mode.values)])
    
        im = plot_homogenous_map(pc, data, ax=hmv_ax)
        hmv_images.append(im)
        
        plot_principal_component(pc=pc, ax=pc_ax)
        plot_pc_seasonality(pc=pc, ax=pc_seas_ax)
        if i==0:
            pc_ax.set_title("Principal Component (PC)", loc='left')
            pc_seas_ax.set_title("PC Seasonality", loc='left')
        # plot_homogenous_map(pc, ax=eof_ax)

    adjust_ticks(fig)
    add_colorbar(eof_images, fig_eof)
    add_colorbar(hmv_images, fig_hmv)

    save(fig, PLOT_DIR/"figure_3.png", add_hash=True, bbox_inches='tight')


def prepare_figure(modes):
    n_modes = len(modes)
    fig = plt.figure(figsize=(10, 13))#, layout='constrained')
    fig_eof, fig_pv, fig_pc = fig.subfigures(3, 1, height_ratios=[5, 5, 7], hspace=0.0)

    fig_eof.subplots(1, n_modes, subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
    fig_pv.subplots(1, n_modes, subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
    fig_pc.subplots(n_modes, 2, width_ratios=[5, 2], height_ratios=[1,]*n_modes, sharey=True)
    fig_pc.subplots_adjust(hspace=.2, wspace=0.05)
    # fig_eof.set_facecolor('#f7d4d4')
    # fig_pc.set_facecolor('#d4eef7')

    for ax in fig_eof.axes + fig_pv.axes:
        ax.polar.add_features(labels=False, 
                            ocean_kwargs={'facecolor': '#EFEFDA'},
                            ruler_kwargs={'primary_color': '#808080', 
                                            'secondary_color': '#EFEFDA'})
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

    # Restrict the domain to the domain north of 50°N
    data = data.sel(lat=slice(50, 90))

    return data


def perform_eof_analysis(data, n_modes=10, plot_expvar=False):
    # Fill NaNs with the mean of the detrended time series (--> 0) to remove variance
    data = data.fillna(data.mean('time'))

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

    if plot_expvar:
        plot_expvar(expvar)
    return rotator
    

def plot_expvar(expvar):
    fig = plt.figure()
    expvar.plot(marker='.', lw=.5, ax=plt.gca())
    plt.title('')
    save(fig, PLOT_DIR/"eof_analysis_explained_variance.png")


def plot_eof_map(eof, ax):
    im = eof.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmocean.cm.balance, add_colorbar=False)
    return im


def add_pval_hatch(pc, data, ax):
    plt.rc('hatch', color='.3', linewidth=.5)

    pval = xarrayutils.xr_linregress(pc, data, dim='time').p_value
    pval.where(pval < .05).plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                            add_colorbar=False,
                                            # levels=[.05,1], lw=2, colors='w'
                                            # )
                                            cmap='none',  # alpha=.7,
                                            hatches=['...'])


def add_homogenous_map_contours(pc, data, ax, levels=[50]):
    lin_reg = xarrayutils.xr_linregress(pc, data, dim='time')
    variance = lin_reg.r_value**2 * 100
    pval = lin_reg.p_value
    CS = variance.plot.contour(
        levels=levels,
        ax=ax,
        colors='cyan',
        linewidths=3,
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=100,  # robust=True,
    )


def plot_principal_component(pc, ax):
    pc.plot(ax=ax, c='k')
    ax.axhline(0, c='.5', lw=1, zorder=0)
    title_dict = {1: 'high-Arctic mode', 2: 'sub-Arctic mode'}
    ax.set_xlabel('')
    ax.set_ylabel(title_dict[int(pc.mode.values)], fontsize='large')
    ax.set_title('')


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


def add_colorbar(images, fig, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.axes as maxes

    for im in images:
        # TODO: Normalize images to the same color range
        ...
    # divider = make_axes_locatable(fig.axes[-1])
    # cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=maxes.Axes)
    
    cax = fig.add_axes([0.95, 0.2, 0.015, 0.6])

    cbar = fig.colorbar(im, cax=cax, orientation='vertical', shrink=0.8, **kwargs)
    # plt.subplots_adjust(wspace=0.05)


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


def plot_homogenous_map(pc, data, ax):
    from xarrayutils import xr_linregress
    res = xr_linregress(data, pc, dim='time')
    pval = res.r_value ** 2 * 100
    im = pval.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
    )
    ax.set_title(f"Homogeneous\nmap of variance")#\n(dof via {dof.replace('_',' ')})")
    return im



if __name__ == "__main__":
    main()

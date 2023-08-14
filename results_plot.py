#!/usr/bin/env -S python3 -i
import sys
import os

import pandas as pd
from typing import Union, Iterable, Optional, Sequence, Callable, List, Any
from collections import OrderedDict
from pathlib import Path
from datetime import datetime as dt
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FixedLocator, FixedFormatter
import seaborn.objects as so
import tikzplotlib
import re

from lib.config import *

# Configure matplotlib
matplotlib.use('TkAgg')


# rename certain columns to make them easier to work with
rename_cols = {
    'Average Latency (microseconds)': 'avg_latency_ms',
    'Throughput (requests/second)': 'throughput',
    'Benchmark Runtime (nanoseconds)': 'total_time_ns',
    'block size': 'block_size',
}


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    Stolen from: https://stackoverflow.com/a/75903189
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
    return obj


def to_mb(ms: str):
    """Convert memory size in postgres config format to # of MB"""
    units = ms[-2:].lower()

    num = float(ms[:-2]) * {
        'mb': 2**0,
        'gb': 2**10,
    }[units]

    return int(num)


def mk_list(x):
    """wrap argument in a list if it isn't one already"""
    if isinstance(x, list):
        return x
    elif isinstance(x, Iterable):
        return list(x)
    else:
        return [x]


def format_str_or_iterable(to_fmt: Union[str, Iterable[str]]) -> str:
    """For a list/sequence of strings, format as comma-separated string"""
    if type(to_fmt) == str:
        return to_fmt

    return ', '.join(str(s) for s in to_fmt)


def format_brnch_ns(to_fmt: Iterable[str]) -> str:
    """
    Renamme PBM branches for the graphs.
    Based on (branch, num_samples)
    """
    to_fmt = list(to_fmt)
    brnch = to_fmt[0]
    samples = str(to_fmt[1])
    if brnch == 'pbm2' and samples == '1':
        return 'Random'
    if brnch == 'pbm3':
        return f'PBM-sampling ({samples}) + freq'

    mapping = {
        'base': 'Clock-sweep',
        'pbm1': 'PBM-PQ',
        'pbm2': 'PBM-sampling',
        'pbm3': 'PBM-sampling + freq',
        'pbm4': 'PBM-sampling + idx',
    }

    if brnch in mapping and samples != '':
        return mapping[brnch] + f' ({samples})'
    elif brnch in mapping:
        return mapping[brnch]
    else:
        return format_str_or_iterable(to_fmt)


fmt_bulk_regex = re.compile('\((\d*)\)')
def format_branch_ns_nv(to_fmt: Iterable[str]) -> str:
    """
    Renamme PBM branches for the graphs.
    Based on (branch, num_samples, num_victims)
    """
    to_fmt = list(to_fmt)
    samples = to_fmt[1]
    victims = to_fmt[2]

    base_fmt = format_brnch_ns(to_fmt)

    # bulk eviction: replace `(# samples)` with `(bulk: # victims/# samples)`
    if to_fmt[0] in ['pbm2', 'pbm3'] and to_fmt[2] not in ['', '1']:
        return base_fmt.replace(to_fmt[1], f'bulk: {victims}/{samples}')
    else:
        return base_fmt


def format_branch_ns_nv_inc(to_fmt: Iterable[str]) -> str:
    """
    Renamme PBM branches for the graphs.
    Based on (branch, num_samples, num_victims, idx_scan_num_counts)
    """
    to_fmt = list(to_fmt)
    base_fmt = format_branch_ns_nv(to_fmt)

    if to_fmt[0] == 'pbm4' and to_fmt[3] not in [None, '', '0']:
        return f'{base_fmt}, idx_counts={to_fmt[3]}'
    else:
        return base_fmt


def to_bool(val: str, default: bool) -> bool:
    true_vals = ['true', 'yes', 't', 'y', 'on']
    false_vals = ['false', 'no', 't', 'f', 'off']
    if not val:
        return default
    elif val.lower() in true_vals:
        return True
    elif val.lower() in false_vals:
        return False
    else:
        raise Exception(f'to_bool unrecognized string: {val}')


def default_fmt_branch(to_fmt: Iterable[str]) -> str:
    """
    Renamme PBM branches for the graphs.
    Based on (branch, num_samples, num_victims, pbm_evict_use_freq, pbm_evict_use_idx_scan, pbm_lru_if_not_requested)
    """
    to_fmt = list(to_fmt)
    base_fmt = format_branch_ns_nv(to_fmt)

    brnch = to_fmt[0]
    samples = str(to_fmt[1])
    use_freq = to_bool(to_fmt[3], False)
    use_idx = to_bool(to_fmt[4], True)
    use_nr_lru = to_bool(to_fmt[5], True)

    ret = base_fmt
    if brnch == 'pbm4':
        ret = f'PBM-sampling ({samples})'
        features = []
        if use_freq:
            features.append('freq')
        if use_idx:
            features.append('idx')
        if use_nr_lru:
            features.append('nrlru')
        if len(features) > 0:
            ret += ' + ' + ', '.join(features)

    return ret


def average_series(df: pd.DataFrame, /, x, y, other_cols=None):
    """
    Compute the average and error for a series.
    'other_cols' are included in the groupby call, should be functionally dependent on 'x'
    """
    if other_cols is None:
        other_cols = []

    other_cols = [c for c in other_cols if c is not None]

    cols = list(set(other_cols + [x, y]))
    grp_cols = list(set(other_cols + [x]))

    df_grps = df[cols].groupby(grp_cols)
    # df_ret = df_grps.agg(['mean', 'std', 'sem'])

    df_ret = df_grps.mean(numeric_only=False)
    df_sem = df_grps.sem(ddof=1)
    df_ret['err'] = df_sem * 1.96

    return df_ret.reset_index()


def plot_exp(df: pd.DataFrame, exp: str, *, ax: Optional[plt.Axes] = None,
             x, xsort=False, xlabels=None, logx=False, xlabel=None,
             y, ylabel=None, ybound=None, avg_y_values=True,
             group: Union[str, Iterable[str]] = ('branch', 'pbm_evict_num_samples'),
             grp_name: Callable[[Iterable[str]], str] = format_str_or_iterable,
             grp_sort: Callable[[Iterable[str]], Any] = None,
             title=None, legend_title=None):
    """Plot an experiment."""
    df_exp = df[df['experiment'].isin(mk_list(exp))]

    if ax is None:
        f, ax = plt.subplots(num=title)

    if type(group) is not str:
        group = list(group)

    # group the data. sort the groups and convert groups to labels for the legend
    grps_plots = list(df_exp.groupby(group))
    if grp_sort is not None:
        grps_plots.sort(key=grp_sort)
    grps_plots = [(grp_name(g), p) for g, p in grps_plots]

    # graph each 'group' as a separate line
    df_plot = None
    for glabel, df_plot in grps_plots:
        # sort x values if requested
        if type(xsort) == bool and xsort:
            xsort = x
        elif xsort is False:
            xsort = None

        # average the y-values for the same x-value first if requested
        if avg_y_values:
            df_plot = average_series(df_plot, x=x, y=y, other_cols=[xsort, xlabels])

        if xsort is not None:
            df_plot = df_plot.sort_values(by=xsort)

        # whether to use log-scale for x or not
        if logx:
            plotfn = lambda *a, **kwa: ax.semilogx(*a, base=2, **kwa)
        else:
            plotfn = ax.plot

        # actually plot the current group
        err_bars = avg_y_values
        if err_bars:
            ebar = ax.errorbar(df_plot[x], df_plot[y], yerr=df_plot['err'], capsize=3)
            # workaround for tikzplotlib: https://github.com/nschloe/tikzplotlib/issues/218#issuecomment-854912145
            # set the labels *after* plotting the series to prevent the error bars themselves being labelled too
            ebar[0].set_label(glabel)
            if logx:
                ax.set_xscale('log')
        else:
            plotfn(df_plot[x], df_plot[y], label=glabel)

    ax.minorticks_off()
    ax.ticklabel_format(useOffset=False)
    if ybound is not None:
        ax.set_ybound(*ybound)
    if xlabels is not None:
        ax.set_xticks(df_plot[x], labels=df_plot[xlabels])

    ax.set_xlabel(xlabel or str(x))
    ax.set_ylabel(ylabel or y)
    ax.legend(title=legend_title or format_str_or_iterable(group))
    ax.set_title(str(title))

    return ax


def bar_plot(df_plot: pd.DataFrame, x, y,):
    grps = df_plot.groupby(x)
    df_plot = pd.DataFrame(grps[y].mean())
    df_plot['yerr'] = grps[y].sem() * 1.96

    return df_plot[y].plot.bar(yerr=df_plot.yerr)


# OBSOLETE: experiment with plotting using seaborn instead of matplotlib directly
def plot_exp_sb(df: pd.DataFrame, exp: str, *, ax: Optional[plt.Axes] = None,
                x, xlabels=None, logx=False, xlabel=None,
                y, ybound=None, ylabel=None,  avg_y_values=False,
                group: Union[str, Sequence[str]] = ('branch', 'pbm_evict_num_samples'),
                title=None):
    df_exp = df[df.experiment == exp]

    if isinstance(group, Sequence):
        df_exp['grouping'] = df_exp[group[0]].astype(str)
        df_exp['grouping'] = df_exp['grouping'].str.cat([df_exp[g] for g in group[1:]], sep=', ', na_rep='').str.strip(', ')
        group = 'grouping'

    pl = so.Plot(df_exp, x=x, y=y, color=group)

    if xlabels is not None:
        # TODO make sure these are in the correct order...
        xvals = list(OrderedDict.fromkeys(df_exp[x]))
        xlabs = list(OrderedDict.fromkeys(df_exp[xlabels]))

        pl = pl.scale(x=so.Continuous()
                        .tick(locator=FixedLocator(xvals))
                        .label(formatter=FixedFormatter(xlabs)))

    # TODO how to log scale?
    # TODO can use "nominal" to work around the issue... (but make sure to sort when passing to the locator)
    # if logx:
    #     pl = pl.scale(x='log')

    if ybound is not None:
        pl = pl.limit(y=ybound)

    pl = pl.add(so.Line(marker='o'), so.Agg()).add(so.Band(), so.Est(errorbar=('se', 1.96)))

    pl.label(title=title, x=xlabel, y=ylabel)

    pl.show()

    ...


# pl_base = so.Plot(df, x='x', y='y', group='g', color='g')
#     pl1 = pl_base.add(so.Line(marker='o'), so.Agg()).add(so.Band(), so.Est(errorbar=('se', 1.96)))
#     pl2 = pl1.scale(x=so.Continuous()
#                       .tick(locator=FixedLocator([1, 2, 3]))
#                       .label(formatter=matplotlib.ticker.FixedFormatter(['a', 'b', 'c']))
#                     )


def post_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate extra columns to be plotted for the given dataframe"""

    # convert hardware stat columns to numeric
    for c in SYSBLOCKSTAT_COLS:
        df[c] = pd.to_numeric(df[c])
    df['hw_read_gb'] = df.sectors_read * 512 / 2**30

    df['hw_mb_per_s'] = df.hw_read_gb * 1024 * 10**9 / df.total_time_ns
    df['pg_mb_per_s'] = df.data_read_gb * 1024 * 10**9 / df.total_time_ns
    df['minutes_total'] = df.total_time_ns / 10**9 / 60
    df['minutes_stream'] = df.max_stream_s / 60

    df['pg_iolat'] = (df.db_blk_read_time / 1000) / (df.db_blks_read * df.block_size / 2**20)  # ms/block to s/GiB
    df['hw_iolat'] = (df.read_ticks / 1000) / (df.sectors_read * 512 / 2**30)  # ms/sectors to s/GiB

    # disk wait time (minutes) (concurrent waits including separate worker threads are added)
    df['pg_disk_wait'] = df.db_blk_read_time / 1000 / 60
    df['hw_disk_wait'] = df.read_ticks / 1000 / 60

    # Convert shared buffers config to numbers for plotting
    df['shmem_mb'] = df['shared_buffers'].map(to_mb)
    df['data_read_per_stream'] = df.data_read_gb / df.parallelism
    df['data_processed_per_stream'] = df.data_processed_gb / df.parallelism
    df['data_read_per_s'] = df.data_read_gb / df.max_stream_s
    df['data_processed_per_s'] = df.data_processed_gb / df.max_stream_s
    df['avg_latency_s'] = df.avg_latency_ms / 1000

    # Calculate separate heap & index hitrate
    df['lineitem_heap_hitrate'] = df['lineitem_heap_blks_hit'] / (df['lineitem_heap_blks_hit'] + df['lineitem_heap_blks_read'])
    df['lineitem_idx_hitrate'] = df['lineitem_idx_blks_hit'] / (df['lineitem_idx_blks_hit'] + df['lineitem_idx_blks_read'])

    return df


def parallelism_grp_sort_key(x: (Iterable[str], Any)):
    # input is tuple (group list, plot)
    g, _ = x
    g = mk_list(g)

    # extract group columns
    brnch, samples, victims, freq, idx, nrlru = g[0:6]
    
    # sort by: branch first, then bulk eviction, # samples (decreasing), other columns are there to get a consistent order but we don't actually care
    return (brnch.lower(), int(victims or '1'), -int(samples or '0'), freq, idx, nrlru)


def plot_figures_parallelism(df: pd.DataFrame, exp: Union[str, list], subtitle: str,
                             hitrate=True, runtime=True, data_processed=False, iorate=False, iolat=False, iovol=False,
                             separate_hitrate=False, time_ybound=(0, None), hitrate_ybound=None,
                             extra_grp_cols=None, grp_name=default_fmt_branch, avg_y_values=True):
    """Generates all the interesting plots for a TPCH parallelism experiment"""
    group_cols = [
        'branch', 'pbm_evict_num_samples', 'pbm_evict_num_victims',
        'pbm_evict_use_freq', 'pbm_evict_use_idx_scan', 'pbm_lru_if_not_requested',
        *(extra_grp_cols or []),
    ]
    parallelism_common_args = {
        'x': 'parallelism', 'xsort': True, 'xlabel': 'Parallelism', 'xlabels': 'parallelism',
        'group': group_cols, 'grp_name': grp_name, 'grp_sort': parallelism_grp_sort_key, 'legend_title': 'Policy',
        'avg_y_values': avg_y_values,
    }
    ret_list = []

    ret_list += [
        plot_exp(df, exp, y='hit_rate', ylabel='Hit rate', ybound=hitrate_ybound,
                 title=f'Hit rate vs parallelism - {subtitle}', **parallelism_common_args),
    ] if hitrate else []

    ret_list += [
        plot_exp(df, exp, y='lineitem_heap_hitrate', ylabel='Heap hit rate', ybound=hitrate_ybound,
                 title=f'Heap hit rate vs parallelism - {subtitle}', **parallelism_common_args),
        plot_exp(df, exp, y='lineitem_idx_hitrate', ylabel='Index hit rate', ybound=hitrate_ybound,
                 title=f'Index hit rate vs parallelism - {subtitle}', **parallelism_common_args),
    ] if separate_hitrate else []

    ret_list += [
        plot_exp(df, exp, y='minutes_total', ylabel='Time (min)', ybound=time_ybound,
                 title=f'Time vs parallelism - {subtitle}', **parallelism_common_args),
    ] if runtime else []

    ret_list += [
        plot_exp(df, exp, y='data_processed_per_stream', ylabel='data_processed',
                 title=f'data processed vs parallelism - {subtitle}', **parallelism_common_args),
    ] if data_processed else []

    ret_list += [
        plot_exp(df, exp, y='data_read_gb', ylabel='I/O volume (GB)', ybound=(0, None),
                 title=f'IO volume vs parallelism - {subtitle}', **parallelism_common_args),
    ] if iovol else []

    ret_list += [
        plot_exp(df, exp, y='pg_mb_per_s', ylabel='IO throughput (MiB/s)', ybound=(0, None),
                 title=f'Postgres IO rate vs parallelism - {subtitle}', **parallelism_common_args),
        plot_exp(df, exp, y='hw_mb_per_s', ylabel='IO throughput (MiB/s)', ybound=(0, None),
                 title=f'Hardware IO rate vs parallelism - {subtitle}', **parallelism_common_args),
    ] if iorate else []

    ret_list += [
        plot_exp(df, exp, y='pg_iolat', ylabel='IO latency (s/GiB)', ybound=(0, None),
                 title=f'Postgres IO latency vs parallelism - {subtitle}', **parallelism_common_args),
        plot_exp(df, exp, y='pg_iolat', ylabel='IO latency (s/GiB)', ybound=(0, None),
                 title=f'Hardware IO latency vs parallelism - {subtitle}', **parallelism_common_args),
    ] if iolat else []

    return ret_list



def plot_figures_shmem(df: pd.DataFrame, exp: Union[str, list], subtitle: str,
                       hitrate=True, runtime=True, data_processed=False, iorate=False, iolat=False, iovol=False,
                       separate_hitrate=False, time_ybound=None, hitrate_ybound=None,
                       extra_grp_cols=None, grp_name=default_fmt_branch, avg_y_values=True):
    """Generates all the interesting plots for a TPCH parallelism experiment"""
    group_cols = [
        'branch', 'pbm_evict_num_samples', 'pbm_evict_num_victims',
        'pbm_evict_use_freq', 'pbm_evict_use_idx_scan', 'pbm_lru_if_not_requested',
        *(extra_grp_cols or []),
    ]
    shmem_common_args = {
        'x': 'shmem_mb', 'xsort': True, 'xlabel': 'Cache size (GiB)', 'xlabels': 'shared_buffers',
        'group': group_cols, 'grp_name': grp_name, 'grp_sort': parallelism_grp_sort_key, 'legend_title': 'Policy',
        'avg_y_values': avg_y_values,
    }
    ret_list = []

    ret_list += [
        plot_exp(df, exp, y='hit_rate', ylabel='Hit rate', ybound=hitrate_ybound,
                 title=f'Hit rate vs cache size - {subtitle}', **shmem_common_args),
    ] if hitrate else []

    ret_list += [
        plot_exp(df, exp, y='lineitem_heap_hitrate', ylabel='Heap hit rate', ybound=hitrate_ybound,
                 title=f'Heap hit rate vs cache size - {subtitle}', **shmem_common_args),
        plot_exp(df, exp, y='lineitem_idx_hitrate', ylabel='Index hit rate', ybound=hitrate_ybound,
                 title=f'Index hit rate vs cache size - {subtitle}', **shmem_common_args),
    ] if separate_hitrate else []

    ret_list += [
        plot_exp(df, exp, y='minutes_total', ylabel='Time (min)', ybound=time_ybound,
                 title=f'Time vs cache size - {subtitle}', **shmem_common_args),
    ] if runtime else []

    ret_list += [
        plot_exp(df, exp, y='data_processed_per_stream', ylabel='data_processed',
                 title=f'data processed vs cache size - {subtitle}', **shmem_common_args),
    ] if data_processed else []

    ret_list += [
        plot_exp(df, exp, y='data_read_gb', ylabel='I/O volume (GB)',
                 title=f'IO volume vs cache size - {subtitle}', **shmem_common_args),
    ] if iovol else []

    ret_list += [
        plot_exp(df, exp, y='pg_mb_per_s', ylabel='IO throughput (MiB/s)',
                 title=f'Postgres IO rate vs cache size - {subtitle}', **shmem_common_args),
        plot_exp(df, exp, y='hw_mb_per_s', ylabel='IO throughput (MiB/s)',
                 title=f'Hardware IO rate vs cache size - {subtitle}', **shmem_common_args),
    ] if iorate else []

    ret_list += [
        plot_exp(df, exp, y='pg_iolat', ylabel='IO latency (s/GiB)',
                 title=f'Postgres IO latency vs cache size - {subtitle}', **shmem_common_args),
        plot_exp(df, exp, y='pg_iolat', ylabel='IO latency (s/GiB)',
                 title=f'Hardware IO latency vs cache size - {subtitle}', **shmem_common_args),
    ] if iolat else []

    return ret_list


def plot_figures_9(df: pd.DataFrame):
    ret_list = [
        *plot_figures_parallelism(df, 'parallelism_cgroup_largeblks_1', '3GB cgroup'),
        *plot_figures_parallelism(df, 'parallelism_nocgroup_1', 'no cgroup', runtime=False, iorate=False, iolat=False),
        *plot_figures_parallelism(df, 'parallelism_nocgroup_largeblks_1', 'no cgroup large blocks', runtime=False, iorate=False, iolat=False),
    ]
    return ret_list


def plot_figures_tpcc(df: pd.DataFrame, exp: str, subtitle: str):
    # tpcc_basic_parallelism, tpcc_basic_parallelism_largeblks_2
    group_cols = ['branch', 'pbm_evict_num_samples', 'pbm_evict_num_victims']
    tpcc_plot_args = {
        'group': group_cols, 'grp_name': format_branch_ns_nv,
        'x': 'parallelism', 'xsort': True, 'xlabel': 'Parallelism', 'xlabels': 'parallelism',
        'avg_y_values': True, 'legend_title': 'Policy',
    }

    res_plots = [
        # hit rate
        plot_exp(df, exp, y='hit_rate', ylabel='Hit rate',
                 title=f'TPCC Hit rate vs parallelism - {subtitle}', **tpcc_plot_args),
        # throughput
        plot_exp(df, exp, y='throughput', ylabel='Throughput',
                 title=f'TPCC Throughput vs parallelism - {subtitle}', **tpcc_plot_args),
        # average latency
        plot_exp(df, exp, y='avg_latency_ms', ylabel='Average Latency (ms)',
                 title=f'TPCC Latency vs parallelism - {subtitle}', **tpcc_plot_args),
    ]
    return res_plots


def create_out_dir() -> Path:
    """Create directory for latex results"""
    while True:
        ts = dt.now()
        ts_str = ts.strftime('%Y-%m-%d_%H-%M')
        res_dir = FIGURES_ROOT / f'{ts_str}'
        try:
            os.makedirs(res_dir)
            return res_dir
        except FileExistsError:
            # if the file already exists, wait and try again with new timestamp
            print(f'WARNING: trying to save results to {res_dir} but it already exists! retrying...')
            time.sleep(15)


def save_plots_as_latex(plots: List[plt.Axes]):
    fig_dir = create_out_dir()
    print(f'Saving plots to {fig_dir}')

    for ax in plots:
        tikzplotlib_fix_ncols(ax.figure)
        tikzplotlib.save(fig_dir / (ax.title.get_text() + '.tikz'), figure=ax.figure, externalize_tables=False)


def main(df: pd.DataFrame, df_old: pd.DataFrame):
    # Output results
    print('================================================================================')
    print('== Post-process interactive prompt:')
    print('==   `df` contains a dataframe of the results')
    print('==   `plt` is `matplotlib.pyplot`')
    print('================================================================================')


    print(f'Generating plots...')

    plots = [
        # *plot_figures_parallelism(df, 'parallelism_cgroup_largeblks_ssd_2', 'SSD + 3GB cgroup'),  # SSD (11)
        # *plot_figures_parallelism(df, 'parallelism_cgroup_smallblks_ssd_2', 'SSD + 3GB cgroup + small blocks'),  # SSD (11)
        # *plot_figures_parallelism(df, 'parallelism_cgroup_sel50_2', '3GB cgroup 50% selectivity', time_ybound=(0, 80)),  # 12
        # *plot_figures_tpcc(df, 'tpcc_basic_parallelism_3', 'HDD small block groups'),
        # *plot_figures_tpcc(df, 'tpcc_basic_parallelism_largeblks_3', 'HDD large block groups'), (empty DF)
        # one of the above failed at some point?
        # *plot_figures_tpcc(df, 'tpcc_basic_parallelism_ssd_3', 'SSD small block groups'),
        # *plot_figures_tpcc(df, 'tpcc_basic_parallelism_largeblks_ssd_3', 'SSD large block groups'),  # failed on second experiment (OOM?) (1 item in DF)
        # *plot_figures_parallelism(df, ['parallelism_idx_ssd_5', 'parallelism_idx_ssd_pbm4_1'], 'index microbenchmarks', separate_hitrate=True),  # index scans (14)
        # *plot_figures_parallelism(df, ['parallelism_idx_ssd__s2_r5_1', 'parallelism_idx_ssd_pbm4_s2_r5_1'], '2% index microbenchmarks',
        #                           separate_hitrate=True, iorate=False, iolat=False),  # 2% index scans (not saved)

        # *plot_figures_parallelism(df, ['parallelism_idx_ssd_no_whole_bg_1'], '2% index microbenchmarks', separate_hitrate=True, iorate=False, iolat=False),  # 1% w/o evict whole group (not saved) -- still no improvement!

        # *plot_figures_parallelism(df_old, ['parallelism_ssd_btree_1'], 'idx + btree', separate_hitrate=False, iorate=False, iolat=False),
    ]

    include_seq_parallel = False
    include_seq_mem = False
    include_seq_hdd = False
    include_seq_ram = False
    include_tpch = True
    include_tpch_brin = False
    include_idx_trailing = False
    include_idx_sequential = False

    ### sequential/bitmap scan microbenchmarks - parallelism - final results
    if include_seq_parallel:
        plots += [
            # compare branches:
            *plot_figures_parallelism(df[df.branch.isin(['base', 'pbm1']) | df.pbm_evict_num_samples.eq('10')],
                                      ['parallelism_micro_seqscans_1'], 'Sequential Scan Microbenchmarks',
                                      iolat=True, iorate=True, iovol=True,),
            # compare sample sizes:
            *plot_figures_parallelism(df[df.branch.isin(['pbm2']) & df.pbm_evict_num_victims.isin(['', '1'])],
                                      ['parallelism_micro_seqscans_1'], 'Sequential Scans - Impact of Sample Size',
                                      iovol=True,),
            # bulk-eviction:
            *plot_figures_parallelism(df[df.branch.isin(['pbm2']) & df.pbm_evict_num_samples.isin(['10', '20', '100'])],
                                      ['parallelism_micro_seqscans_1'], 'Sequential Scans - Impact of Bulk Eviction',
                                      iovol=True),
        ]

    ### sequential/bitmap scan microbenchmarks - memory - final results
    if include_seq_mem:
        plots += [
            # compare branches:
            *plot_figures_shmem(df[df.branch.isin(['base', 'pbm1']) | df.pbm_evict_num_samples.eq('10')],
                                ['shmem_micro_seqs_1'], 'Sequential Scan Microbenchmarks',
                                iovol=True,),
            # compare sample sizes:
            *plot_figures_shmem(df[df.branch.isin(['pbm2'])],
                                ['shmem_micro_seqs_1'], 'Sequential Scans - Impact of Sample Size',
                                iovol=True,),
        ]

    # HDD sequential experiments:
    if include_seq_hdd:
        plots += plot_figures_parallelism(df[df.branch.isin(['base', 'pbm1']) | df.pbm_evict_num_samples.eq('10')],
                                          ['parallelism_micro_seqscans_hdd_1'], 'HDD Sequential Scan Microbenchmarks',
                                          iolat=True, iorate=True, iovol=True,)

    # RAM sequential experiments:
    if include_seq_ram:
        plots += plot_figures_parallelism(df[df.branch.isin(['base', 'pbm1']) | df.pbm_evict_num_samples.eq('10')],
                                          ['parallelism_micro_seqscans_ram_1'], 'RAM Sequential Scan Microbenchmarks',
                                          iovol=True,)

    if include_tpch:
        # plots += plot_figures_parallelism(df, ['tpch_3', 'tpch_pbm4_1_all'], 'TPCH', iolat=True, iorate=True, iovol=True,)
        plots += plot_figures_parallelism(df[df.pbm_evict_num_samples.isin(['20', ''])], ['tpch_3', 'tpch_pbm4_1_all'], 'TPCH', iolat=False, iorate=False, iovol=True,)
        # plots += plot_figures_parallelism(df[df.pbm_evict_num_samples.isin(['10', ''])], ['tpch_3', 'tpch_pbm4_1_all'], 'TPCH', iolat=False, iorate=False, iovol=True,)


    if include_tpch_brin:
        plots += plot_figures_parallelism(df, ['tpch_brin_3',], 'TPCH BRIN only', iolat=True, iorate=True, iovol=True)

    if include_idx_trailing:
        plots += [
            ### trailing index scan microbenchmarks - final results
            *plot_figures_parallelism(df, ['micro_idx_parallelism_baseline_1'], 'Trailing index scans 1pct',
                                      separate_hitrate=False, iorate=False, iolat=False, iovol=True),
            # *plot_figures_parallelism(df, ['micro_idx_parallelism_baseline_1', 'micro_idx_parallelism_pbm4_1'], 'Trailing index scans 1pct', separate_hitrate=False, iorate=False, iolat=False),
            *plot_figures_parallelism(df, ['micro_idx_parallelism_pbm4_2_idx', 'micro_idx_parallelism_pbm4_2_idx+lru_nr', 'micro_idx_parallelism_pbm4_2_no_idx+lru_nr', 'micro_idx_parallelism_pbm4_2_freq+lru_nr', 'micro_idx_parallelism_pbm4_2_all'],
                                      'Trailing index scans 1pct EXTRA', separate_hitrate=False, iorate=False, iolat=False, iovol=True),
            # ^ RESULTS: no_idx+lru_nr is good. Conclusions: frequency is good, idx support does nothing, lru_nr doesn't help without frequency stats...

            *plot_figures_parallelism(df, ['micro_idx_parallelism_baseline_1', 'micro_idx_parallelism_high_baseline_1', 'micro_idx_parallelism_pbm4_2_idx', 'micro_idx_parallelism_high_pbm4_2_idx'], 'Trailing index MORE EXTRAS',
                                      separate_hitrate=False, iorate=False, iolat=False, iovol=False),
            # ^ going to higher parallelism, index doesn't actually seem to help unfortunately

        ]

    if include_idx_sequential:
        plots += [
            ### sequential index scan microbenchmarks - final results
            *plot_figures_parallelism(df, ['parallelism_ssd_btree_1'], 'Sequential index scans',
                                      separate_hitrate=False, iorate=False, iolat=False, iovol=True,
                                      time_ybound=(0, 60), hitrate_ybound=(0.9, 1.0)),
            # *plot_figures_parallelism(df, ['parallelism_ssd_btree_1', 'parallelism_ssd_btree_pbm4_2'], 'Sequential index scans', separate_hitrate=False, iorate=False, iolat=False,
            #                           time_ybound=(0, 100), hitrate_ybound=(0.9, 1.0)),
            *plot_figures_parallelism(df, ['parallelism_ssd_btree_pbm4_3_idx', 'parallelism_ssd_btree_pbm4_3_idx+lru_nr', 'parallelism_ssd_btree_pbm4_3_no_idx+lru_nr', 'parallelism_ssd_btree_pbm4_3_freq+lru_nr', 'parallelism_ssd_btree_pbm4_3_all'],
                                      'Sequential index scans EXTRAS', separate_hitrate=False, iorate=False, iolat=False, iovol=True,
                                      time_ybound=(0, 60), hitrate_ybound=(0.9, 1.0)),
            # TODO NOTE!: the above 'micro_idx_parallelism_pbm4_1' and 'parallelism_ssd_btree_pbm4_2' do NOT have nr_lru enabled, auto-generated legend is not correct.

        ]

    # TODO testing...
    # plots += [
    #     *plot_figures_parallelism(#df,
    #         df[df.branch.isin(['pbm2']) & df.pbm_evict_num_victims.isin(['', '1'])],
    #                               #df[df.branch.isin(['base', 'pbm1']) | df.pbm_evict_num_samples.eq('10')],
    #                               ['parallelism_micro_seqscans_1'], 'testing',),
    # ]





    return df, plots


"""
Main entry point: read the results csv file (including old ones), compute some extra columns, plot results,
and then do some more processing for manual examination/debugging.
"""
if __name__ == '__main__':
    # Read in the data, converting certain column names
    args = sys.argv[1:]

    df = pd.read_csv(COLLECTED_RESULTS_CSV, keep_default_na=False).rename(columns=rename_cols)
    df_orig = df.copy()

    # include some results from the old set of experiments
    old_experiments_to_include = [
        # 'parallelism_idx_ssd_5',  # trailing index benchmarks with non-PBM4 branches
    ]

    # old experiments
    other_dfs = []
    for old_res in os.listdir('old_results'):
        other_dfs.append(pd.read_csv('old_results/' + old_res, keep_default_na=False).rename(columns=rename_cols))
    df_old = pd.concat(other_dfs, ignore_index=True)

    df = pd.concat([df, df_old[df_old.experiment.isin(old_experiments_to_include)]], ignore_index=True)

    # Add some columns for convenience/plotting
    df = post_process_data(df)
    df_old = post_process_data(df_old)

    # generate the plots
    df, plots = main(df, df_old)

    # decide what to do with the plots
    def save_plots():
        save_plots_as_latex(plots)

    def show_plots():
        print(f'Showing plots...')
        plt.show()

    for arg in args:
        arg = arg.lower()
        if arg == 'save':
            save_plots()
        elif arg in ['plot', 'show']:
            show_plots()


# TODO clean this up, was some manual data analysis after generating the plots

    group_cols = [
        'experiment', 'branch', 'block_size', 'block_group_size',
        # 'pbm_evict_num_samples',
        # 'pbm_bg_naest_max_age'
    ]

    df_g = df[[
        *group_cols,

        'hit_rate', 'minutes_total', 'minutes_stream', 'pg_mb_per_s', 'hw_mb_per_s',
        # 'data_read_gb',
        # postgres DB stats
        'db_active_time', 'db_blk_read_time', 'db_blks_hit', 'db_blks_read',
        # index vs heap blks:
        'heap_blks_hit', 'heap_blks_read', 'idx_blks_hit', 'idx_blks_read',
        # other HW io stats
        *SYSBLOCKSTAT_COLS,
        # calculated:
        'pg_iolat', 'hw_iolat', 'pg_disk_wait', 'hw_disk_wait',
    ]]
    # g = df_g.groupby(group_cols)
    # res = g.mean()
    # res['min_t_ci'] = g['minutes_total'].sem() * 1.96
    # res['min_s_ci'] = g['minutes_stream'].sem() * 1.96
    # res['hit_rate_ci'] = g['hit_rate'].sem() * 1.96

    # cg = df[df.experiment == 'test_cgroup']

    # iostat_cols = ['pg_iolat', 'hw_iolat', 'pg_disk_wait', 'hw_disk_wait']

    # df['heap_total'] = df.lineitem_heap_blks_hit + df.lineitem_heap_blks_read
    # df['idx_total'] = df.lineitem_idx_blks_hit + df.lineitem_idx_blks_read
    # df['pct_idx'] = 100 * df.idx_total / (df.idx_total + df.heap_total)



# (Manual) Post-processing of the final graphs included in the paper:
#  - Comment-out titles, use captions instead
#  - Set ymin, ymax if appropriate
#  - Change legend position if it isn't placed well (especially after adjusting the bounds)
#  - Parallelism graphs: add `xtick=data,` for the axis labels
#  - Parallelism graphs: maybe remove the x=1 label?
#  - Cache size: replace `xticklabels={,,1GB,2GB,3GB,4GB,5GB,6GB,7GB,8GB},` (remove label for <1GB) to prevent labels from overlapping
#  - for experiments with different sample sizes: re-order stuff to have the sample sizes sorted
#  - for plots with bulk eviction: `#` character needs to be changed



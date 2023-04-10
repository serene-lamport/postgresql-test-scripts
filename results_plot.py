#!/usr/bin/env -S python3 -i
import pandas as pd
from typing import Union, Iterable, Optional, Sequence, Callable
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FixedLocator, FixedFormatter
import seaborn.objects as so

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


def to_mb(ms: str):
    """Convert memory size in postgres config format to # of MB"""
    units = ms[-2:].lower()

    num = float(ms[:-2]) * {
        'mb': 2**0,
        'gb': 2**10,
    }[units]

    return int(num)


def format_str_or_iterable(to_fmt: Union[str, Iterable[str]]) -> str:
    """For a list/sequence of strings, format as comma-separated string"""
    if type(to_fmt) == str:
        return to_fmt

    return ', '.join(str(s) for s in to_fmt)


def format_brnch_ns(to_fmt: Iterable[str]) -> str:
    """Renamme PBM branches for the graphs."""
    to_fmt = list(to_fmt)
    if to_fmt[0] == 'pbm2' and str(to_fmt[1]) == '1':
        return 'Random'

    mapping = {
        'base': 'GCLOCK',
        'pbm1': 'PBM-PQ',
        'pbm2': 'PBM-sampling',
        'pbm3': 'pbm3',  # TODO what to call this?
        # TODO other branches?
    }

    if to_fmt[0] in mapping:
        return mapping[to_fmt[0]]
    else:
        return format_str_or_iterable(to_fmt)

def format_branch_ns_nv(to_fmt: Iterable[str]) -> str:
    to_fmt = list(to_fmt)
    if to_fmt[0] == 'pbm2' and str(to_fmt[1]) == '1':
        return 'Random'

    base_fmt = format_brnch_ns(to_fmt)

    if to_fmt[0] in ['pbm2', 'pbm3'] and to_fmt[2] not in ['', '1']:
        return f'{base_fmt}, # evicted={to_fmt[2]}'
    else:
        return base_fmt



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
             title=None, legend_title=None):
    """Plot an experiment."""
    df_exp = df[df['experiment'] == exp]

    if ax is None:
        f, ax = plt.subplots(num=title)

    if type(group) is not str:
        group = list(group)

    df_plot = None
    for grp, df_plot in df_exp.groupby(group):
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
            ax.errorbar(df_plot[x], df_plot[y], yerr=df_plot['err'], label=grp_name(grp), capsize=3)
            if logx:
                ax.set_xscale('log')
            # plotfn(df_plot[x], df_plot[y], label=grp_name(grp), yerr=df_plot['err'])
        else:
            plotfn(df_plot[x], df_plot[y], label=grp_name(grp))

    ax.minorticks_off()
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


def add_reads(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


def plot_old_results(df: pd.DataFrame):
    # # plot some experiments
    plots = [
        # plot_exp (df, 'test_reset_stats_shmem',
        #                  x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
        #                  y='hit_rate', ylabel='hit rate', ybound=(0,1),
        #                  title='Hit-rate vs shared buffer size'),
        # plot_exp(df, 'test_shmem_prewarm_2', group=['branch', 'prewarm'],
        #                  x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
        #                  y='hit_rate', ylabel='hit rate', ybound=(0,1),
        #                  title='Hit-rate vs shared buffer size with and without pre-warming'),
        # plot_exp(df, 'test_sf100', group=['branch', 'prewarm'],
        #                  x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
        #                  y='hit_rate', ylabel='hit rate', ybound=(0,1),
        #                  title='SF 100 Hit-rate vs shared buffer size'),

        # plot_exp(df, 'test_scripts_buffer_sizes', group=['branch', 'pbm_evict_num_samples'],
        #                  x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
        #                  y='hit_rate', ylabel='hit rate', ybound=(0,1),
        #                  title='Hit-rate vs shared buffer'),

        # microbenchmarks varying the buffer size, with and without prewarm
        # plot_exp(df[df.prewarm == True], 'buffer_sizes_4', group=['branch', 'pbm_evict_num_samples'],
        # plot_exp(df[(df.prewarm == True) & (df.parallelism == 8)], 'buffer_sizes_4',
        #          group=['branch', 'pbm_evict_num_samples'],
        #          x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
        #          y='hit_rate', ylabel='hit rate', ybound=(0,1), avg_y_values=True,
        #          title='Hit-rate vs shared buffer size with (averaged, parallelism = 8)'),
        # plot_exp(df[(df.prewarm == True) & (df.parallelism == 16)], 'buffer_sizes_4',
        #          group=['branch', 'pbm_evict_num_samples'],
        #          x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
        #          y='hit_rate', ylabel='hit rate', ybound=(0,1), avg_y_values=True,
        #          title='Hit-rate vs shared buffer size with (averaged, parallelism = 16)'),
        # plot_exp(df[df.prewarm == False], 'buffer_sizes_4', group=['branch', 'pbm_evict_num_samples'],
        # plot_exp(df[(df.prewarm == False)], 'buffer_sizes_4', group=['branch', 'pbm_evict_num_samples'],
        #          x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
        #          y='hit_rate', ylabel='hit rate', ybound=(0,1), avg_y_values=True,
        #          title='Hit-rate vs shared buffer size without pre-warming'),

        # microbenchmaks varying parallelism with and without syncscans
        # plot_exp(df[df.synchronize_seqscans == 'on'], 'parallelism_3', group=['branch', 'pbm_evict_num_samples'],
        #          x='parallelism', xsort=True, xlabel='Parallelism',
        #          y='hit_rate', ylabel='hit rate', ybound=(0,1),
        #          title='Hit-rate vs parallelism with syncscans'),
        # # plot_exp(df[df.synchronize_seqscans == 'off'], 'parallelism_3', group=['branch', 'pbm_evict_num_samples'],
        # #          x='parallelism', xsort=True, xlabel='Parallelism',
        # #          y='hit_rate', ylabel='hit rate', ybound=(0,1),
        # #          title='Hit-rate vs parallelism without syncscans'),
        # plot_exp(df, 'parallelism_same_nqueries_1', group=['branch', 'pbm_evict_num_samples'],
        #          x='parallelism', xsort=True, xlabel='Parallelism',
        #          y='hit_rate', ylabel='hit rate', ybound=(0, 1),
        #          title='Hit-rate vs parallelism with syncscans'),
        # plot_exp(df, 'parallelism_same_nqueries_1', group=['branch', 'pbm_evict_num_samples'],
        #          x='parallelism', xsort=True, xlabel='Parallelism',
        #          y='throughput',
        #          title='Throughput vs parallelism'),
        # plot_exp(df, 'parallelism_same_nqueries_1', group=['branch', 'pbm_evict_num_samples'],
        #          x='parallelism', xsort=True, xlabel='Parallelism',
        #          y='data_per_stream', ylabel='Data read (GiB)',
        #          title='Data volume (per stream) vs parallelism'),



        # parallelism_same_nqueries_1,   parallelism_3

    ]

    # f, axs = plt.subplots(2, 2)
    # for i, sel in enumerate([20, 40, 60, 80]):
    #     plot_exp(df, f'parallelism_sel{sel}_2', ax=axs[i//2][i%2], group=['branch', 'pbm_evict_num_samples'],
    #              x='parallelism', xsort=True, xlabel='Parallelism',
    #              y='hit_rate', ylabel='hit rate', ybound=(0, 1),
    #              title=f'{sel}% selectivity hit_rate vs parallelism')

    # plot_exp(df, 'test_weird_spike_1', group=['branch', 'pbm_evict_num_samples'],
    #          x='parallelism', xsort=True, xlabel='Parallelism',
    #          y='hit_rate', ylabel='hit rate', ybound=(0, 1), avg_y_values=True,
    #          title='averaged hit_rate vs parallelism 1')


    # hit-rate as a function of buffer size for different configurations
    # for nsamples in ['2', '5', '10', '20']:
    # for nsamples in ['5', '10', '20']:
    #     f, ax = plt.subplots(1, 2)
    #     df_filtered = df[df.pbm_evict_num_samples.isin(['', '1', nsamples]) & (df.prewarm == True)]
    #     plot_exp(df_filtered, 'buffer_sizes_4', ax=ax[0],
    #              group=['branch', 'pbm_evict_num_samples'],
    #              x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
    #              y='hit_rate', ylabel='hit rate', ybound=(0,1), avg_y_values=True,
    #              title=f'parallelism = 8, samples = {nsamples}'),
    #     plot_exp(df_filtered, 'buffer_sizes_p16_1', ax=ax[1],
    #              group=['branch', 'pbm_evict_num_samples'],
    #              x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
    #              y='hit_rate', ylabel='hit rate', ybound=(0,1), avg_y_values=True,
    #              title=f'parallelism = 16, samples = {nsamples}'),
    #     f.suptitle(f'Hit-rate vs shared buffer size (samples = {nsamples})')
    #
    # for nsamples in ['5', '10']:
    #     f, ax = plt.subplots(1, 2)
    #     df_filtered = df[df.pbm_evict_num_samples.isin(['', '1', nsamples]) & (df.prewarm == True)]
    #     plot_exp(df_filtered, 'buffer_sizes_4', ax=ax[0],
    #              group=['branch', 'pbm_evict_num_samples'],
    #              x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
    #              y='average_stream_s', ylabel='Average stream time', avg_y_values=True,
    #              title=f'parallelism = 8, samples = {nsamples}'),
    #     plot_exp(df_filtered, 'buffer_sizes_p16_1', ax=ax[1],
    #              group=['branch', 'pbm_evict_num_samples'],
    #              x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
    #              y='average_stream_s', ylabel='Average stream time', avg_y_values=True,
    #              title=f'parallelism = 16, samples = {nsamples}'),
    #     f.suptitle(f'Average stream time vs shared buffer size (samples = {nsamples})')
    #
    # for nsamples in ['5', '10']:
    #     df_filtered = df[df.pbm_evict_num_samples.isin(['', '1', nsamples])]
    #     plot_exp(df_filtered, 'test_weird_spike_3', group=['branch', 'pbm_evict_num_samples'],
    #              x='parallelism', xsort=True, xlabel='Parallelism',
    #              y='hit_rate', ylabel='hit rate', ybound=(0, 1), avg_y_values=True,
    #              title=f'averaged hit_rate vs parallelism (samples = {nsamples})')
    #     # plot_exp_sb(df_filtered, 'test_weird_spike_3', group=['branch', 'pbm_evict_num_samples'],
    #     #             x='parallelism', xlabel='Parallelism',  # xsort=True,
    #     #             y='hit_rate', ylabel='hit rate', ybound=(0, 1), avg_y_values=True,
    #     #             title=f'averaged hit_rate vs parallelism')
    #
    # for nsamples in ['2', '5', '10']:
    #     df_filtered = df[df.pbm_evict_num_samples.isin(['', '1', nsamples])]
    #     plot_exp(df_filtered, 'test_weird_spike_3', group=['branch', 'pbm_evict_num_samples'],
    #              x='parallelism', xsort=True, xlabel='Parallelism',
    #              y='average_stream_s', ylabel='Average stream time', avg_y_values=True,
    #              title=f'Average stream time vs parallelism (samples = {nsamples})')
    #     plot_exp(df_filtered, 'test_weird_spike_3', group=['branch', 'pbm_evict_num_samples'],
    #              x='parallelism', xsort=True, xlabel='Parallelism',
    #              y='max_stream_s', ylabel='Max stream time', avg_y_values=True,
    #              title=f'Max stream time vs parallelism (samples = {nsamples})')
    # TODO redo ^ using `parallelism_sel30_1` - should be less random...


    # TODO \/ are the parallelism experiments showing clock being better... but only 1 query per stream!
    # plot_exp(df, 'parallelism_2', group=['branch', 'pbm_evict_num_samples'],
    #          x='parallelism', xsort=True, xlabel='Parallelism',  # logx=True,
    #          y='hit_rate', ylabel='hit_rate', avg_y_values=True,
    #          title=f'Hit rate vs parallelism')
    #
    # plot_exp(df, 'parallelism_2', group=['branch', 'pbm_evict_num_samples'],
    #          x='parallelism', xsort=True, xlabel='Parallelism',  # logx=True,
    #          y='minutes_total', ylabel='Time (min)', avg_y_values=True,
    #          title=f'Time vs parallelism')



    #
    # f, axs = plt.subplots(2, 3)
    # for i, s in enumerate([16312, 22289, 16987, 6262, 32495, 5786]):
    #     plot_exp(df[df.seed == s], 'test_weird_spike_3', ax=axs[i//3][i%3],
    #              group=['branch', 'pbm_evict_num_samples'],
    #              x='parallelism', xsort=True, xlabel='Parallelism',
    #              y='hit_rate', ylabel='hit rate', ybound=(0, 1),
    #              title=f'hit_rate vs parallelism seed={s}')


    # data processed/read per second/stream
    # for nsamples in ['5', '10']:
    # for nsamples in ['10']:
    for nsamples in []:
        df_filtered = df[df.pbm_evict_num_samples.isin(['', '1', nsamples]) & (df.branch != 'pbm3')]
        plot_exp(df_filtered, 'test_weird_spike_3', group=['branch', 'pbm_evict_num_samples'],
                 x='parallelism', xsort=True, xlabel='Parallelism',
                 y='data_read_per_stream', ylabel='Data read/stream (GB)', avg_y_values=True,
                 title=f'Data read per stream vs parallelism (samples = {nsamples})')
        plot_exp(df_filtered, 'test_weird_spike_3', group=['branch', 'pbm_evict_num_samples'],
                 x='parallelism', xsort=True, xlabel='Parallelism',
                 y='data_processed_per_stream', ylabel='Data processed/stream (GB)', avg_y_values=True,
                 title=f'Data processed per stream vs parallelism (samples = {nsamples})')
        plot_exp(df_filtered, 'test_weird_spike_3', group=['branch', 'pbm_evict_num_samples'],
                 x='parallelism', xsort=True, xlabel='Parallelism',
                 y='data_read_per_s', ylabel='Data read/s (GB)', avg_y_values=True,
                 title=f'Data read per second vs parallelism (samples = {nsamples})')
        plot_exp(df_filtered, 'test_weird_spike_3', group=['branch', 'pbm_evict_num_samples'],
                 x='parallelism', xsort=True, xlabel='Parallelism',
                 y='data_processed_per_s', ylabel='Data processed/s (GB)', avg_y_values=True,
                 title=f'Data processed per second vs parallelism (samples = {nsamples})')


def plot_figures_parallelism(df: pd.DataFrame, exp: str, subtitle: str, hitrate=True, runtime=True, data_processed=False, iorate=True, iolat=True,
                             time_ybound=None):
    """Generates all the interesting plots for a TPCH parallelism experiment"""
    group_cols = ['branch', 'pbm_evict_num_samples', 'pbm_evict_num_victims']
    parallelism_common_args = {
        'x': 'parallelism', 'xsort': True, 'xlabel': 'Parallelism', 'xlabels': 'parallelism',
        'group': group_cols, 'grp_name': format_branch_ns_nv, 'legend_title': 'Policy',
        'avg_y_values': True,
    }
    ret_list = []

    ret_list += [
        plot_exp(df, exp, y='hit_rate', ylabel='Hit rate',
                 title=f'Hit rate vs parallelism - {subtitle}', **parallelism_common_args),
    ] if hitrate else []

    ret_list += [
        plot_exp(df, exp, y='minutes_total', ylabel='Time (min)', ybound=time_ybound,
                 title=f'Time vs parallelism - {subtitle}', **parallelism_common_args),
    ] if runtime else []

    ret_list += [
        plot_exp(df, exp, y='data_processed_per_stream', ylabel='data_processed',
                 title=f'data processed vs parallelism - {subtitle}', **parallelism_common_args),
    ] if data_processed else []

    ret_list += [
        plot_exp(df, exp, y='pg_mb_per_s', ylabel='IO throughput (MiB/s)',
                 title=f'Postgres IO rate vs parallelism - {subtitle}', **parallelism_common_args),
        plot_exp(df, exp, y='hw_mb_per_s', ylabel='IO throughput (MiB/s)',
                 title=f'Hardware IO rate vs parallelism - {subtitle}', **parallelism_common_args),
    ] if iorate else []

    ret_list += [
        plot_exp(df, exp, y='pg_iolat', ylabel='IO latency (s/GiB)',
                 title=f'Postgres IO latency vs parallelism - {subtitle}', **parallelism_common_args),
        plot_exp(df, exp, y='pg_iolat', ylabel='IO latency (s/GiB)',
                 title=f'Hardware IO latency vs parallelism - {subtitle}', **parallelism_common_args),
    ] if iolat else []

    return ret_list


def plot_figures_9(df: pd.DataFrame):
    ret_list = [
        *plot_figures_parallelism(df, 'parallelism_cgroup_largeblks_1', '3GB cgroup'),
        *plot_figures_parallelism(df, 'parallelism_nocgroup_1', 'no cgroup', runtime=False, iorate=False, iolat=False),
        *plot_figures_parallelism(df, 'parallelism_nocgroup_largeblks_1', 'no cgroup large blocks', runtime=False, iorate=False, iolat=False),
    ]
    return ret_list


def plot_figures_10_tpcc(df: pd.DataFrame):
    # tpcc_basic_parallelism, tpcc_basic_parallelism_largeblks_2
    group_cols = ['branch', 'pbm_evict_num_samples']
    tpcc_plot_args = {
        'group': group_cols, 'grp_name': format_branch_ns_nv,
        'x': 'parallelism', 'xsort': True, 'xlabel': 'Parallelism', 'xlabels': 'parallelism',
        'avg_y_values': True,
    }
    plots_hit_rate = lambda: [
        plot_exp(df, 'tpcc_basic_parallelism', y='hit_rate', ylabel='Hit rate',
                 title=f'TPCC Hit rate vs parallelism - small block groups', **tpcc_plot_args),
        plot_exp(df, 'tpcc_basic_parallelism_largeblks_2', y='hit_rate', ylabel='Hit rate',
                 title=f'TPCC Hit rate vs parallelism - large block groups', **tpcc_plot_args),
    ]

    plots_throughput = lambda: [
        plot_exp(df, 'tpcc_basic_parallelism', y='throughput', ylabel='Throughput',
                 title=f'TPCC Throughput vs parallelism - small block groups', **tpcc_plot_args),
        plot_exp(df, 'tpcc_basic_parallelism_largeblks_2', y='throughput', ylabel='Throughput',
                 title=f'TPCC Throughput vs parallelism - large block groups', **tpcc_plot_args),

    ]

    res_plots = [
        *plots_hit_rate(),
        *plots_throughput(),
    ]
    return res_plots



def main(df: pd.DataFrame):
    # Output results
    print('================================================================================')
    print('== Post-process interactive prompt:')
    print('==   `df` contains a dataframe of the results')
    print('==   `plt` is `matplotlib.pyplot`')
    print('================================================================================')

    print(f'Generating plots...')

    plots = [
        # *plot_figures_9(df),
        # *plot_figures_10_tpcc(df),
        # *plot_figures_11_ssd(df),
        *plot_figures_parallelism(df, 'parallelism_cgroup_largeblks_ssd_2', 'SSD + 3GB cgroup'),  # SSD (11)
        *plot_figures_parallelism(df, 'parallelism_cgroup_smallblks_ssd_2', 'SSD + 3GB cgroup + small blocks'),  # SSD (11)
        # *plot_figures_parallelism(df, 'parallelism_cgroup_sel50_2', '3GB cgroup 50% selectivity', time_ybound=(0, 80)),  # 12
    ]

    print(f'Showing plots...')
    plt.show()

    return df, plots



if __name__ == '__main__':
    # Read in the data, converting certain column names
    df = pd.read_csv(COLLECTED_RESULTS_CSV, keep_default_na=False).rename(columns=rename_cols)
    df_orig = df.copy()

    # Add some columns for convenience

    # Convert shared buffers config to numbers for plotting
    df['shmem_mb'] = df['shared_buffers'].map(to_mb)
    df['data_read_per_stream'] = df.data_read_gb / df.parallelism
    df['data_processed_per_stream'] = df.data_processed_gb / df.parallelism
    df['data_read_per_s'] = df.data_read_gb / df.max_stream_s
    df['data_processed_per_s'] = df.data_processed_gb / df.max_stream_s
    df = add_reads(df)

    df, plots = main(df)


    # shmem_at_sf100_with_iostats
    # shmem_at_sf100_with_caching
    # shmem_at_sf100_with_caching_2
    # shmem_at_sf100_with_caching_more_iostats

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
        # other HW io stats
        *SYSBLOCKSTAT_COLS,
        # calculated:
        'pg_iolat', 'hw_iolat', 'pg_disk_wait', 'hw_disk_wait',
    ]]
    g = df_g.groupby(group_cols)
    res = g.mean()
    res['min_t_ci'] = g['minutes_total'].sem() * 1.96
    res['min_s_ci'] = g['minutes_stream'].sem() * 1.96
    res['hit_rate_ci'] = g['hit_rate'].sem() * 1.96

    # e1 = 'comparing_bg_lock_types'
    # e2 = 'comparing_bg_lock_types_2'
    # e3 = 'comparing_bg_lock_types_3'
    # e4 = 'comparing_bg_lock_types_4'
    #
    # es = [e1, e2, e3, e4]

    cg = df[df.experiment == 'test_cgroup']

    # e1 = 'shmem_at_sf100_with_caching_more_iostats'
    # e2 = 'shmem_at_sf100_with_caching_more_iostats_bs32'

    e = 'shmem_at_sf100_group_eviction'
    e4096 = 'shmem_at_sf100_group_eviction_bgsz4096'

    f = 'sampling_overhead'
    g = 'sampling_overhead_2'

    e1 = 'shmem_at_sf100_multi_evict_nv1'
    e10 = 'shmem_at_sf100_multi_evict_nv10'

    iostat_cols = ['pg_iolat', 'hw_iolat', 'pg_disk_wait', 'hw_disk_wait']


    # TODO comparing multi eviction at sf 10 instead of 100 --- try again with the cgroup this time :/
    e_sf10 = ['shmem_multi_evict_sf10_nv1_2', 'shmem_multi_evict_sf10_nv10_2']

    # # TODO plot parallelism with cgroup limits (should look like before!) ... this graph has weird results ...
    # e_parallel = 'parallelism_2'

    # TODO plot TPCC, and at larger block sizes
    e_tpcc = 'tpcc_basic_parallelism'


    # TODO bar charts!
    # res.loc[f].hit_rate.plot.bar(yerr=res.loc[f].hit_rate_ci)
    # res.minutes_total.plot.bar(yerr=res.min_t_ci)
    # plt.show()

    # bar_plot(df[df.experiment == 'sampling_overhead'], x=['branch', 'pbm_evict_num_samples'], y='hit_rate')
    # plt.show()



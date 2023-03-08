#!/usr/bin/env -S python3 -i
import pandas as pd
from typing import Union, Iterable, Optional, Sequence
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
             y, ylabel=None, ybound=None, avg_y_values=False,
             group: Union[str, Iterable[str]] = ('branch', 'pbm_evict_num_samples'),
             title=None):
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
            ax.errorbar(df_plot[x], df_plot[y], yerr=df_plot['err'], label=format_str_or_iterable(grp))
            if logx:
                ax.set_xscale('log')
            # plotfn(df_plot[x], df_plot[y], label=format_str_or_iterable(grp), yerr=df_plot['err'])
        else:
            plotfn(df_plot[x], df_plot[y], label=format_str_or_iterable(grp))

    ax.minorticks_off()
    if ybound is not None:
        ax.set_ybound(*ybound)
    if xlabels is not None:
        ax.set_xticks(df_plot[x], labels=df_plot[xlabels])

    ax.set_xlabel(xlabel or str(x))
    ax.set_ylabel(ylabel or y)
    ax.legend(title=format_str_or_iterable(group))
    ax.set_title(str(title))

    return ax


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



def main(df: pd.DataFrame):
    # Output results
    print('================================================================================')
    print('== Post-process interactive prompt:')
    print('==   `df` contains a dataframe of the results')
    print('==   `plt` is `matplotlib.pyplot`')
    print('================================================================================')

    print(f'Generating plots...')

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






    print(f'Showing plots...')
    plt.show()

    return df, plots

    # TODO consider `seaborn` package for better visualization...


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


    df_g = df[[
        'experiment', 'branch', 'block_size', 'pbm_evict_num_samples', 'pbm_bg_naest_max_age',
        'hit_rate', 'minutes_total', 'minutes_stream', 'pg_mb_per_s', 'hw_mb_per_s',
        # 'data_read_gb',
        # postgres DB stats
        'db_active_time', 'db_blk_read_time', 'db_blks_hit', 'db_blks_read',
        # other HW io stats
        *SYSBLOCKSTAT_COLS,
        # calculated:
        'pg_iolat', 'hw_iolat', 'pg_disk_wait', 'hw_disk_wait',
    ]]
    g = df_g.groupby(['experiment', 'branch', 'block_size', 'pbm_evict_num_samples', 'pbm_bg_naest_max_age'])
    res = g.mean()
    res['min_t_ci'] = g['minutes_total'].sem() * 1.96
    res['min_s_ci'] = g['minutes_stream'].sem() * 1.96
    res['hit_rate_ci'] = g['hit_rate'].sem() * 1.96

    e1 = 'comparing_bg_lock_types'
    e2 = 'comparing_bg_lock_types_2'
    e3 = 'comparing_bg_lock_types_3'
    e4 = 'comparing_bg_lock_types_4'

    es = [e1, e2, e3, e4]

    cg = df[df.experiment == 'test_cgroup']

    e = 'shmem_at_sf100_with_caching_more_iostats'



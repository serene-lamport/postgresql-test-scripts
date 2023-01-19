#!/usr/bin/env -S python3 -i
import pandas as pd
from typing import Union, Iterable
import matplotlib.pyplot as plt
import matplotlib

from lib.config import *

# Configure matplotlib
matplotlib.use('TkAgg')


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


def plot_exp(df: pd.DataFrame, exp: str,
             x, xsort=None, xlabels=None, logx=False, xlabel=None,
             y='hit_rate',  ybound=None, ylabel=None,
             group: Union[str, Iterable[str]] = 'branch',
             title='plot'):
    """Plot an experiment."""
    df_exp = df[df['experiment'] == exp]

    f, ax = plt.subplots()

    for grp, df_plot in df_exp.groupby(group):
        if type(xsort) == bool and xsort:
            xsort = x
        if xsort is not None and xsort is not False:
            df_plot = df_plot.sort_values(by=xsort)

        if logx:
            plotfn = lambda *a, **kwa: ax.semilogx(*a, base=2, **kwa)
        else:
            plotfn = ax.plot

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

    return f, ax


if __name__ == '__main__':
    # Read in the data
    df = pd.read_csv(COLLECTED_RESULTS_CSV, keep_default_na=False)

    # Convert shared buffers config to numbers for plotting
    df['shmem_mb'] = df['shared_buffers'].map(to_mb)

    print('================================================================================')
    print('== Post-process interactive prompt:')
    print('==   `df` contains a dataframe of the results')
    print('==   `plt` is `matplotlib.pyplot`')
    print('================================================================================')

    print(f'Generating plots...')

    # # plot some experiments
    # f, ax = plot_exp(df, 'test_reset_stats_shmem',
    #                  x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
    #                  y='hit_rate', ylabel='hit rate', ybound=(0,1),
    #                  title='Hit-rate vs shared buffer size')
    # f, ax = plot_exp(df, 'test_shmem_prewarm_2', group=['branch', 'prewarm'],
    #                  x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
    #                  y='hit_rate', ylabel='hit rate', ybound=(0,1),
    #                  title='Hit-rate vs shared buffer size with and without pre-warming')
    # f, ax = plot_exp(df, 'test_sf100', group=['branch', 'prewarm'],
    #                  x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
    #                  y='hit_rate', ylabel='hit rate', ybound=(0,1),
    #                  title='SF 100 Hit-rate vs shared buffer size')

    f, ax = plot_exp(df, 'test_scripts_buffer_sizes', group=['branch', 'pbm_evict_num_samples'],
                     x='shmem_mb', xsort=True, xlabels='shared_buffers', logx=True, xlabel='shared memory',
                     y='hit_rate', ylabel='hit rate', ybound=(0,1),
                     title='Hit-rate vs shared buffer')

    print(f'Showing plots...')
    plt.show()

    # TODO consider `seaborn` package for better visualization...
    # TODO split script into generating results.csv (collect_results?) and processing results.csv (graphing, etc.)

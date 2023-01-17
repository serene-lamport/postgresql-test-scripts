#!/usr/bin/env -S python3 -i

from pathlib import Path
import os
import json
import csv
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Iterable


###################
#  CONFIGURATION  #
###################
# (these must be the same as main.py)
# (TODO move to a shared file?)

BUILD_ROOT = Path('/home/ta3vande/PG_TESTS')
RESULTS_ROOT = BUILD_ROOT / 'results'

CONFIG_FILE_NAME = 'test_config.json'
CONSTRAINTS_FILE = 'constraints.csv'
INDEXES_FILE = 'indexes.csv'

NON_DIR_RESULTS = [CONFIG_FILE_NAME, CONSTRAINTS_FILE, INDEXES_FILE]


#################
#  CSV columns  #
#################

statio_main_cols = ['heap_blks_hit', 'heap_blks_read', 'idx_blks_hit', 'idx_blks_read']
statio_toast_cols = ['tidx_blks_hit', 'tidx_blks_read', 'toast_blks_hit', 'toast_blks_read']
bbase_latency_cols = [
    'Average Latency (microseconds)',
    'Maximum Latency (microseconds)',
    '99th Percentile Latency (microseconds)',
    '95th Percentile Latency (microseconds)',
    '90th Percentile Latency (microseconds)',
    '75th Percentile Latency (microseconds)',
    'Median Latency (microseconds)',
    '25th Percentile Latency (microseconds)',
    'Minimum Latency (microseconds)',
]

csv_cols = [
    # configuration from directory information:
    'experiment', 'dir', 'branch', 'block size',
    # configuration from configuration json file
    'block_group_size', 'workload', 'scalefactor', 'clustering', 'indexes', 'shared_buffers', 'work_mem',
    'synchronize_seqscans', 'pbm_evict_num_samples', 'parallelism', 'time', 'count_multiplier', 'prewarm',
    # from benchbase summary:
    'Throughput (requests/second)', 'Goodput (requests/second)', 'Benchmark Runtime (nanoseconds)',
    # latency from benchbase summary:
    *bbase_latency_cols,
    # stats from metrics
    'average_stream_s',
    *statio_main_cols,
    *('lineitem_' + col for col in statio_main_cols),
    'hit_rate', 'lineitem_hit_rate',
    'data_read_gb', 'data_processed_gb'
]


##########
#  CODE  #
##########


def read_config(config_dir) -> Optional[dict]:
    """Read and parse the config file, or return `None` if it isn't valid."""
    path = RESULTS_ROOT / config_dir / CONFIG_FILE_NAME
    try:
        with open(path, 'r') as f:
            contents = f.read()

        if not contents:
            return None
        return json.JSONDecoder().decode(contents)

    except (FileNotFoundError, NotADirectoryError):
        return None


def io_metrics_map(metrics: dict, blk_sz: int) -> dict:
    """Parse the `metrics.json` json file produced by benchbase and extract io metrics."""
    pg_statio_user_tables = metrics['pg_statio_user_tables']

    metrics_totals = {}

    for col in statio_main_cols:
        metrics_totals[col] = sum(int(r[col] or 0) for r in pg_statio_user_tables)

    # We're ignoring stats for toast tables assuming they are 0. Check that assumption holds.
    for col in statio_toast_cols:
        s = sum(int(r[col] or 0) for r in pg_statio_user_tables)
        if s > 0:
            print(f'WARNING: found non-zero toast values in column {col}!')

    # Compute stats for just lineitem
    for col in statio_main_cols:
        metrics_totals['lineitem_' + col] = sum(int(r[col] or 0) for r in pg_statio_user_tables if r['relname'] == 'lineitem')

    # Compute hit-rate. (arguably not necessary, can post-process in excel too)
    total_hits = metrics_totals['heap_blks_hit'] + metrics_totals['idx_blks_hit']
    total_reads = metrics_totals['heap_blks_read'] + metrics_totals['idx_blks_read']
    lineitem_total_hits = metrics_totals['lineitem_heap_blks_hit'] + metrics_totals['lineitem_idx_blks_hit']
    lineitem_total_reads = metrics_totals['lineitem_heap_blks_read'] + metrics_totals['lineitem_idx_blks_read']

    metrics_totals['hit_rate'] = total_hits / (total_hits + total_reads)
    metrics_totals['lineitem_hit_rate'] = lineitem_total_hits / (lineitem_total_hits + lineitem_total_reads)
    # Compute amount of data read/processed
    # blk_sz is in KiB, not bytes, so conversion rate is 2^20 to get to GiB
    metrics_totals['data_read_gb'] = total_reads * blk_sz / (2**20)
    metrics_totals['data_processed_gb'] = (total_reads + total_hits) * blk_sz / (2**20)

    return metrics_totals


def to_mb(ms: str):
    """Convert memory size in postgres config format to # of MB"""
    units = ms[-2:].lower()

    num = float(ms[:-2]) * {
        'mb': 2**0,
        'gb': 2**10,
    }[units]

    return int(num)


def format_str_or_iterable(to_fmt: Union[str, Iterable[str]]) -> str:
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

    decoder = json.JSONDecoder()
    json_decode = decoder.decode

    out = open('results.csv', 'w')
    writer = csv.DictWriter(out, csv_cols, extrasaction='ignore')
    writer.writeheader()

    rows = []

    # Process everythign in the results directory
    conf_dir: str
    for conf_dir in os.listdir(RESULTS_ROOT):

        # read config file if it is there
        config = read_config(conf_dir)

        if config is None:
            print(f'{conf_dir}: No config, skipping...')
            continue

        # each directory is a different run of benchbase
        subdirs = [subdir for subdir in os.listdir(RESULTS_ROOT / conf_dir) if subdir not in NON_DIR_RESULTS]
        pgconfigs = [subdir.split('_blksz') for subdir in subdirs]
        try:
            pgconfigs = [(s[0], int(s[1])) for s in pgconfigs]
        except (ValueError, IndexError) as e:
            print(f'ERROR: some subdirectory did not match the naming scheme, don\'t know how to parse {conf_dir}!')
            print(f'    {e!r}')

        try:
            for brnch, blk_sz in pgconfigs:
                subdir = RESULTS_ROOT / conf_dir / f'{brnch}_blksz{blk_sz}'

                # Process benchbase output:
                with open(subdir / 'metrics.json', 'r') as metrics_file:
                    metrics = json_decode(metrics_file.read())
                with open(subdir / 'summary.json', 'r') as summary_file:
                    summary = json_decode(summary_file.read())
                try:
                    with open(subdir / 'stream_times.json', 'r') as st_file:
                        stream_times = json_decode(st_file.read())
                except FileNotFoundError:
                    stream_times = []

                io_metrics = io_metrics_map(metrics, blk_sz)

                # generate row in the processed results:
                row = {
                    'dir': conf_dir,
                    'branch': brnch,
                    'block size': blk_sz,
                    # compute average stream time (stream times are microseconds)
                    'average_stream_s': sum(stream_times) / len(stream_times) / (10**6) if stream_times else None,
                    **config,
                    **summary,
                    **summary['Latency Distribution'],
                    **io_metrics,
                }

                rows.append(row)

                writer.writerow(row)

        except FileNotFoundError as e:
            print(f'ERROR: could not find the benchbase files for {conf_dir}!')
            print(f'    {e!r}')

    out.close()

    df = pd.DataFrame(rows)

    print('================================================================================')
    print('== Post-process interactive prompt:')
    print('==   `df` contains a dataframe of the results')
    print('==   `plt` is `matplotlib.pyplot`')
    print('================================================================================')

    # Plot things
    import matplotlib
    matplotlib.use('TkAgg')

    # Convert shared buffers config to numbers for plotting
    df['shmem_mb'] = df['shared_buffers'].map(to_mb)

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

    plt.show()

    # TODO consider `seaborn` package for better visualization...
    # TODO split script into generating results.csv (collect_results?) and processing results.csv (graphing, etc.)

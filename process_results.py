#!/usr/bin/env python3

from pathlib import Path
import os
import json
import csv
from typing import Optional


###################
#  CONFIGURATION  #
###################
# (these must be the same as main.py)
# (TODO move to a shared file?)

BUILD_ROOT = Path('/home/ta3vande/PG_TESTS')
RESULTS_ROOT = BUILD_ROOT / 'results'

CONFIG_FILE_NAME = 'test_config.json'


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
    'block_group_size', 'workload', 'scalefactor', 'clustering', 'indexes', 'shared_buffers', 'synchronize_seqscans',
    'parallelism', 'time',
    # from benchbase summary:
    'Throughput (requests/second)', 'Goodput (requests/second)',
    # latency from benchbase summary:
    *bbase_latency_cols,
    # stats from metrics
    *statio_main_cols,
    *('lineitem_' + col for col in statio_main_cols),
    'hit_rate', 'lineitem_hit_rate',
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


def io_metrics_map(metrics: dict) -> dict:
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

    return metrics_totals


if __name__ == '__main__':

    decoder = json.JSONDecoder()
    json_decode = decoder.decode

    out = open('results.csv', 'w')
    writer = csv.DictWriter(out, csv_cols, extrasaction='ignore')
    writer.writeheader()

    # Process everythign in the results directory
    conf_dir: str
    for conf_dir in os.listdir(RESULTS_ROOT):

        # read config file if it is there
        config = read_config(conf_dir)

        if config is None:
            print(f'{conf_dir}: No config, skipping...')
            continue

        # print(f'{conf_dir}: {config}')

        # each directory is a different run of benchbase

        pgconfigs = [subdir.split('_blksz') for subdir in os.listdir(RESULTS_ROOT / conf_dir) if subdir != CONFIG_FILE_NAME]
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

                io_metrics = io_metrics_map(metrics)

                # generate row in the processed results:
                row = {
                    'dir': conf_dir,
                    'branch': brnch,
                    'block size': blk_sz,
                    **config,
                    **summary,
                    **summary['Latency Distribution'],
                    **io_metrics,
                }

                writer.writerow(row)

        except FileNotFoundError as e:
            print(f'ERROR: could not find the benchbase files for {conf_dir}!')
            print(f'    {e!r}')





    #
    # total_heap_blks_hit = sum(int(r['heap_blks_hit'] or 0) for r in pg_statio_user_tables)
    # total_heap_blks_read = sum(int(r['heap_blks_read'] or 0) for r in pg_statio_user_tables)
    # total_idx_blks_hit = sum(int(r['idx_blks_hit'] or 0) for r in pg_statio_user_tables)
    # total_idx_blks_read = sum(int(r['idx_blks_read'] or 0) for r in pg_statio_user_tables)
    # total_tidx_blks_hit = sum(int(r['tidx_blks_hit'] or 0) for r in pg_statio_user_tables)
    # total_tidx_blks_read = sum(int(r['tidx_blks_read'] or 0) for r in pg_statio_user_tables)
    # total_toast_blks_hit = sum(int(r['toast_blks_hit'] or 0) for r in pg_statio_user_tables)
    # total_toast_blks_read = sum(int(r['toast_blks_read'] or 0) for r in pg_statio_user_tables)
    #
    #
    # print(f'heap hit: {total_heap_blks_hit:,}, heap read: {total_heap_blks_read:,}')
    # print(f'idx hit: {total_idx_blks_hit:,}, idx read: {total_idx_blks_read:,}')
    # print(f'tidx hit: {total_tidx_blks_hit}, tidx read: {total_tidx_blks_read}')
    # print(f'toast hit: {total_toast_blks_hit}, toast read: {total_toast_blks_read}')
    #
    # lineitem_heap_blks_hit = sum(int(r['heap_blks_hit'] or 0) for r in pg_statio_user_tables if r['relname'] == 'lineitem')
    # lineitem_heap_blks_read = sum(int(r['heap_blks_read'] or 0) for r in pg_statio_user_tables if r['relname'] == 'lineitem')
    # lineitem_idx_blks_hit = sum(int(r['idx_blks_hit'] or 0) for r in pg_statio_user_tables if r['relname'] == 'lineitem')
    # lineitem_idx_blks_read = sum(int(r['idx_blks_read'] or 0) for r in pg_statio_user_tables if r['relname'] == 'lineitem')
    #
    # print(f'lineitem heap hit: {lineitem_heap_blks_hit:,}, lineitem heap read: {lineitem_heap_blks_read:,}')
    # print(f'lineitem idx hit: {lineitem_idx_blks_hit:,}, lineitem idx read: {lineitem_idx_blks_read:,}')
    # heap_hit_rate = total_heap_blks_hit / (total_heap_blks_hit + total_heap_blks_read)
    # linitem_hit_rate = lineitem_heap_blks_hit / (lineitem_heap_blks_hit + lineitem_heap_blks_read)
    #
    # print(f'total: {heap_hit_rate}, lineitem: {linitem_hit_rate}')


    # TODO:
    # move this summing into a function or something and add it to the CSV for each experiment.


    # metrics_encoded = json.JSONEncoder(indent=2).encode(pg_statio_user_tables)
    # print(metrics_encoded)


# TODO: crawl the results directory and create a csv row for each... put interesting data in a CSV file.

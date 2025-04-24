import os 
import json 
from pathlib import Path 
import pandas as pd 
from lib.config import * 
import sys 
import csv 
from typing import List, Dict, Callable

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_blk_read_time(row: dict) -> float:
    if 'pg_stat_database.TPCH_50.blk_read_time' in row: 
        return row['pg_stat_database.TPCH_50.blk_read_time']
    elif 'pg_stat_database.TPCH_20.blk_read_time' in row: 
        return row['pg_stat_database.TPCH_20.blk_read_time']
    return -1
    
    
def collect(res_subdir: Path) -> Dict[str, dict]: 
    
    with open(res_subdir / 'results.json', 'r') as fp: 
        res = json.load(fp)
        
    interesting_keys = {
        'branch': 'config.branch', 
        'scalefactor': 'config.scalefactor',
        'label': 'config.label',
        'experiment_label': 'config.experiment',
        'shared_buffers': 'config.shared_buffers',
        'shared_buffers_gb': 'config.shared_buffers_gb',
        'samples': 'config.pbm_evict_num_samples',
        'seed': 'config.seed',
        'selectivity': 'config.selectivity',
        'parallelism': 'config.parallelism',
        'read_ios': 'iostats.difference.read_ios',
        'write_ios': 'iostats.difference.write_ios',
        'hit_rate': 'pg_statio_user_tables.lineitem.hit_rate', 
        'heap_blks_hit': 'pg_statio_user_tables.lineitem.heap_blks_hit',
        'heap_blks_read': 'pg_statio_user_tables.lineitem.heap_blks_read',
        'idx_blks_hit': 'pg_statio_user_tables.lineitem.idx_blks_hit',
        'idx_blks_read': 'pg_statio_user_tables.lineitem.idx_blks_read',
        # 'blk_read_time': 'pg_stat_database.TPCH_50.blk_read_time',
        'total_io_read_blks': 'pg_statio_user_tables.lineitem.total_io_read_blks',
        'total_io_read_bytes': 'pg_statio_user_tables.lineitem.total_io_read_bytes',
        'total_read_blks': 'pg_statio_user_tables.lineitem.total_read_blks',
        "total_read_bytes": 'pg_statio_user_tables.lineitem.total_read_bytes',
        'total_hit_blks': 'pg_statio_user_tables.lineitem.total_hit_blks',
        'total_miss_blks': 'pg_statio_user_tables.lineitem.total_miss_blks',
        'runtime': 'bbase_summary.Benchmark Runtime (nanoseconds)'
    }
    
    computed_keys = {
        'io_read_gb': lambda row: row['total_io_read_bytes'] / 2**30,
        'runtime': lambda row: row['runtime'] / 1e9,
        'gb_per_min': lambda row: row['io_read_gb'] / (row['runtime'] / 60), 
        'runtime_min': lambda row: row['runtime'] / 60,
        'blk_read_time': lambda row: get_blk_read_time(row),
    }
        
    out = []
    for expkey, expval in res.items(): 
        cur_exp = {}
        expval = flatten_dict(expval)
        cur_exp['expkey'] = expkey
        for k, flatkey in interesting_keys.items():
            val = expval[flatkey]
            cur_exp[k] = val
            
        for k, f in computed_keys.items():
            cur_exp[k] = f(cur_exp)
            
        out.append(cur_exp)
    
    out = sorted(out, key=lambda x: f"{x['branch']}_{str(x['samples']).zfill(4)}_{str(x['parallelism']).zfill(4)}_{x['expkey']}")
    with open(res_subdir / 'results.csv', 'w') as fp: 
        writer = csv.DictWriter(fp, fieldnames=['expkey'] + list(interesting_keys.keys()) + list(computed_keys.keys()))
        writer.writeheader()
        writer.writerows(out)
        

        
        
    
    



if __name__ == '__main__':
    res_dir = Path(RESULTS_ROOT) 
    res_subdir = res_dir / sys.argv[1]
    collect(res_subdir)
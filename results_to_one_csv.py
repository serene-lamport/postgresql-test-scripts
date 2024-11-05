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

def collect(res_subdir: Path) -> Dict[str, dict]: 
    
    with open(res_subdir / 'results.json', 'r') as fp: 
        res = json.load(fp)
        
    interesting_keys = {
        'branch': 'config.branch', 
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
        'blk_read_time': 'pg_stat_database.TPCH_100.blk_read_time',
    }
        
    out = []
    for expkey, expval in res.items(): 
        cur_exp = {}
        expval = flatten_dict(expval)
        for k, flatkey in interesting_keys.items():
            val = expval[flatkey]
            cur_exp[k] = val
            
        out.append(cur_exp)
        
    with open(res_subdir / 'results.csv', 'w') as fp: 
        writer = csv.DictWriter(fp, fieldnames=interesting_keys.keys())
        writer.writeheader()
        writer.writerows(out)
        

        
        
    
    



if __name__ == '__main__':
    res_dir = Path(RESULTS_ROOT) 
    res_subdir = res_dir / sys.argv[1]
    collect(res_subdir)
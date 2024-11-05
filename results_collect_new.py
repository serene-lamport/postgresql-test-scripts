

import os 
import json 

from pathlib import Path 
import pandas as pd 
from lib.config import * 
import sys 
import csv 
from typing import List

def read_json_file(file_path: Path):
    with open(file_path) as f: 
        return json.load(f)

def read_csv_file(file_path: Path):
    with open(file_path, 'r') as fp:
        rows = list(csv.DictReader(fp))
    return rows
    
def read_iostats(exp_dir: Path): 
    iostats = read_json_file(exp_dir / 'iostats.json')

    res = dict()
    for k, v in iostats['after'].items(): 
        res[k] = float(v) - float(iostats['before'][k])
    return res


def calculate_hit_rate(db_stats: List[dict], statio_user_tables: List[dict]):
    for row in statio_user_tables: 
        if row['relname'] == 'lineitem': 
            lineitem = row
            break
        
    total_hits = int(lineitem['heap_blks_hit']) + int(lineitem['idx_blks_hit'])
    total_reads = int(lineitem['heap_blks_read']) + int(lineitem['idx_blks_read'])
    
    hit_rate = total_hits / (total_hits + total_reads)
    
    return hit_rate


def get_computed_properties(basic_info: List[dict], iostats: dict, db_stats: List[dict], statio_user_tables: List[dict]):
    res = dict()
    
    # hit rate
    hit_rate = calculate_hit_rate(db_stats, statio_user_tables)
    res['hit_rate'] = hit_rate
    
    
    return res
    
    
def collect(res_subdir: Path): 
    res = []
    exp_dirs = os.listdir(res_subdir)
    for exp_dir in exp_dirs: 
        exp_dir = res_subdir / exp_dir
        
        # Get basic information
        basic_info = read_json_file(exp_dir / 'test_config.json')
        
        # Calculate iostats
        iostats = read_iostats(exp_dir)
        
        # Get database stats
        db_stats = read_csv_file(exp_dir / 'pg_stat_database.csv')
        
        # Get statio user tables
        statio_user_tables = read_csv_file(exp_dir / 'pg_statio_user_tables.csv')
        
        # get computed properties 
        computed = get_computed_properties(basic_info, iostats, db_stats, statio_user_tables)
        res += [
            {
                **basic_info, 
                **iostats, 
                **computed
            }
        ]
        
    with open(res_subdir / 'results.csv', 'w') as f: 
        writer = csv.DictWriter(f, fieldnames=res[0].keys())
        writer.writeheader()
        writer.writerows(res)
        
        
        
        
        
        
        
        
         
    


if __name__ == '__main__': 
    res_dir = RESULTS_ROOT
    res_dir = Path(res_dir)
    
    res_subdir = res_dir / sys.argv[1]
    
    collect(res_subdir)
    
import os 
import json 
from pathlib import Path 
import pandas as pd 
from lib.config import * 
import sys 
import csv 
from typing import List, Dict, Callable


# What do we have in the output of each run?
# 1. Experiment configuration in 'test_config.json'
# 2. IO stats in 'iostats.json'
# 3. A bunch of csv files from pg_stats: 
#     - pg_stat_archiver.csv
#     - pg_stat_bgwriter.csv
#     - pg_stat_database.csv
#     - pg_stat_database_conflicts.csv
#     - pg_stat_user_tables.csv
#     - pg_statio_user_tables.csv
#     - pg_stat_user_indexes.csv
#     - pg_statio_user_indexes.csv



# This file is supposed to consolidate all the results from each run into a single dictionary
# This file is NOT supposed to make any computations, it just collects the data and spits it out
# The computations will be done by plotter scripts


def read_json_file(file: Path, lambdas: Dict[str, Callable]=dict()) -> dict:
    with open(file, 'r') as fp: 
        res = json.load(fp)
    
    for k, v in lambdas.items(): 
        res[k] = v(res)
        
    return res


def read_csv_file(file: Path, group_by: str, lambdas: Dict[str, Callable]=dict()) -> dict: 
    with open(file, 'r') as fp: 
        rows = list(csv.DictReader(fp))
    
    res = dict()
    for row in rows: 
        
        # correct the types for the rows so lambdas can work 
        for k, v in row.items(): 
            try: 
                row[k] = int(v)
            except ValueError: 
                try: 
                    row[k] = float(v)
                except ValueError: 
                    if v == '': 
                        row[k] = 0
                    else: 
                        print(f"Could not convert {v} to int or float")
                
        # do the grouping
        key = row[group_by]
        res[key] = row
        
        # perform the lambdas
        for k, v in lambdas.items():
            try: 
                res[key][k] = v(row)
            except ZeroDivisionError: 
                res[key][k] = 0.0
            
    return res

def get_label(branch: str, samples: int) -> str: 
    if branch == 'base': 
        return "Clock Sweep"
    if branch == 'pbm1': 
        return "PQ"
    if branch == 'pbm2': 
        return f"PBM {samples}"
    else:
        return"TODOFROMCONSOLIDATE"
    
def collect(res_subdir: Path) -> Dict[str, dict]: 
    
    res = dict()
    
    # only get the directories in res_subdir and ignore the files:
    exp_dirs = [d for d in os.listdir(res_subdir) if os.path.isdir(res_subdir / d)]
    
    kb = 1024
    mb = 1024 * kb
    gb = 1024 * mb
    blk_size = 8 * kb    
    
    for exp_dir in exp_dirs: 
        exp_id = str(exp_dir)
        exp_dir = res_subdir / exp_dir
        exp_res = {
            "config": read_json_file(
                exp_dir / 'test_config.json', 
                lambdas={
                    "label": lambda data: get_label(data['branch'], data['pbm_evict_num_samples'])
                }
            ),
            "iostats": read_json_file(
                exp_dir / 'iostats.json', 
                lambdas={
                    "difference": lambda data: {k: float(data['after'][k]) - float(data['before'][k]) for k in data['after'].keys()}
                }
            ), 
            "pg_stat_database": read_csv_file(
                exp_dir / 'pg_stat_database.csv', 
                'datname', 
                lambdas={
                    "hit_rate": lambda row: row["blks_hit"] / (row["blks_hit"] + row["blks_read"])
                }
            ),
            "pg_stat_user_indexes": read_csv_file(exp_dir / 'pg_stat_user_indexes.csv', 'indexrelname'),
            "pg_stat_user_tables": read_csv_file(exp_dir / 'pg_stat_user_tables.csv', 'relname'),
            "pg_statio_user_indexes": read_csv_file(
                exp_dir / 'pg_statio_user_indexes.csv', 
                'indexrelname', 
                lambdas={
                    "hit_rate": lambda row: row["idx_blks_hit"] / (row["idx_blks_hit"] + row["idx_blks_read"]), 
                    "tot_idx_blks": lambda row: row["idx_blks_hit"] + row["idx_blks_read"]
                }
            ),
            "pg_statio_user_tables": read_csv_file(
                exp_dir / 'pg_statio_user_tables.csv', 
                'relname', 
                lambdas={
                    "hit_rate": lambda row: row["heap_blks_hit"] / (row["heap_blks_hit"] + row["heap_blks_read"]),
                    "total_io_read_blks": lambda row: row["heap_blks_read"] + row["idx_blks_read"],
                    "total_io_read_bytes": lambda row: row["total_io_read_blks"] * blk_size,
                    "total_read_blks": lambda row: row["total_io_read_blks"] + row["idx_blks_hit"] + row["heap_blks_hit"],
                    "total_read_bytes": lambda row: row["total_read_blks"] * blk_size,
                    "total_hit_blks": lambda row: row["idx_blks_hit"] + row["heap_blks_hit"],
                    "total_miss_blks": lambda row: row["total_read_blks"] - row["total_hit_blks"],
                }
            )
            
        }
        
        # print(exp_res)
        for key in exp_res.keys(): 
            print(key)
            for kk in exp_res[key].keys(): 
                print("    ", kk)
        res[exp_id] = exp_res
    
    
    with open(res_subdir / 'results.json', 'w') as fp: 
        json.dump(res, fp)
    return res
        
        
    
    



if __name__ == '__main__':
    res_dir = Path(RESULTS_ROOT) 
    res_subdir = res_dir / sys.argv[1]
    collect(res_subdir)
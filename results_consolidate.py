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
                    pass
                
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

def collect(res_subdir: Path) -> Dict[str, dict]: 
    
    res = dict()
    
    # only get the directories in res_subdir and ignore the files:
    exp_dirs = [d for d in os.listdir(res_subdir) if os.path.isdir(res_subdir / d)]
    
    for exp_dir in exp_dirs: 
        exp_id = str(exp_dir)
        exp_dir = res_subdir / exp_dir
        exp_res = {
            "config": read_json_file(exp_dir / 'test_config.json'),
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
                    "tot_heap_blks": lambda row: row["heap_blks_hit"] + row["heap_blks_read"], 
                    "tot_idx_blks": lambda row: row["idx_blks_hit"] + row["idx_blks_read"]
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
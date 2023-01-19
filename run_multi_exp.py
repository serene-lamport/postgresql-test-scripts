#!/usr/bin/env python3
from dataclasses import replace
from typing import Iterable
from datetime import datetime as dt
from itertools import product

from lib.experiments import *


###################
#   HELPER CODE   #
###################


def run_tests(exp_name: str, tests: Iterable[ExperimentConfig], /, dry_run=False):
    """
    Run a set of experiments.
    """
    tests = list(tests)
    count = len(tests)
    c_len = len(str(count))
    for i, exp in enumerate(tests):
        start = dt.now()
        ts_str = start.strftime('%H:%M:%S')

        print_str = f'===     STARTING EXPERIMENT  {exp_name} #{i+1:{c_len}}/{count}  at  {ts_str}     ==='

        print('='*len(print_str))
        print(print_str)
        print('='*len(print_str))

        if dry_run:
            print(f'EXPERIMENT: {exp.dbconf = }  {exp.pgconf = }')
        else:
            run_experiment(exp_name, exp)

        end = dt.now()
        ts_str = end.strftime('%H:%M:%S')
        elapsed = (end - start)
        e_min = elapsed.seconds // 60
        e_s = elapsed.total_seconds() % 60
        print_str = f'===     END EXPERIMENT  {exp_name} #{i+1:{c_len}}/{count}  at  {ts_str}  ({e_min}m {e_s:.1f}s)     ==='

        print('='*len(print_str))
        print(print_str)
        print('='*len(print_str))
        print()


def samples(brnch: PgBranch, ns: List[int]):
    return ns if brnch.accepts_nsamples else [None]


##############################
#   EXPERIMENT DEFINITIONS   #
##############################


def _TEST_test_buffer_sizes() -> Iterable[ExperimentConfig]:
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')
    bbconf = BBaseConfig(nworkers=2, workload=WORKLOAD_MICRO_COUNTS.with_multiplier(2))

    base_dbconf = DbConfig(branch=BRANCH_POSTGRES_BASE, sf=1)

    for shmem, brnch in product(['128MB', '1GB', '4GB'], POSTGRES_ALL_BRANCHES):
        dbconf = replace(base_dbconf, branch=brnch)

        for nsamples in samples(brnch, [1, 10]):
            pgconf = RuntimePgConfig(shared_buffers=shmem, pbm_evict_num_samples=nsamples)

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf)


def test_micro_shared_memory() -> Iterable[ExperimentConfig]:
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')
    base_bbconf = BBaseConfig(nworkers=8, workload=WORKLOAD_MICRO_COUNTS.with_multiplier(8))
    base_dbconf = DbConfig(branch=BRANCH_POSTGRES_BASE, sf=10)

    shmem_ops = ['256MB', '512MB', '1GB', '2GB', '4GB', '8GB', '16GB']

    for shmem, prewarm, branch in product(shmem_ops, [True, False], POSTGRES_ALL_BRANCHES):
        dbconf = replace(base_dbconf, branch=branch)
        bbconf = replace(base_bbconf, prewarm=prewarm)

        for nsamples in samples(branch, [1, 2, 5, 10, 20]):
            pgconf = RuntimePgConfig(shared_buffers=shmem, pbm_evict_num_samples=nsamples)

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf)


def test_micro_parallelism() -> Iterable[ExperimentConfig]:
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')
    base_dbconf = DbConfig(branch=BRANCH_POSTGRES_BASE, sf=10)

    shmem = '2GB'
    total_queries = 2**6
    parallel_ops = [1, 2, 4, 8, 16, 32]
    syncscan_ops = ['on', 'off']

    for nworkers, branch in product(parallel_ops, POSTGRES_ALL_BRANCHES):
        cm = total_queries // nworkers

        dbconf = replace(base_dbconf, branch=branch)
        bbconf = BBaseConfig(nworkers=nworkers, workload=WORKLOAD_MICRO_COUNTS.with_multiplier(cm))

        for nsamples, syncscans in product(samples(branch, [1, 2, 5, 10, 20]), syncscan_ops):
            pgconf = RuntimePgConfig(shared_buffers=shmem,
                                     pbm_evict_num_samples=nsamples,
                                     synchronize_seqscans=syncscans)

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf)


if __name__ == '__main__':
    # Run actual experiments
    # run_tests('test_scripts_buffer_sizes_2', _TEST_test_buffer_sizes())
    run_tests('buffer_sizes_1', test_micro_shared_memory())
    run_tests('parallelism_1', test_micro_parallelism())

    ...

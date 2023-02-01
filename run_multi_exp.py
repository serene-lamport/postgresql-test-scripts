#!/usr/bin/env python3
from dataclasses import replace
from typing import Iterable
from datetime import datetime as dt
from itertools import product

from lib.experiments import *


###################
#   HELPER CODE   #
###################

# GLOBALS
NUM_EXPERIMENTS_RUN: int = 0


def run_tests(exp_name: str, tests: Iterable[ExperimentConfig], /, dry_run=False):
    """
    Run a set of experiments.
    """
    global NUM_EXPERIMENTS_RUN
    tests = list(tests)
    count = len(tests)
    c_len = len(str(count))
    for i, exp in enumerate(tests):
        start = dt.now()
        ts_str = start.strftime('%H:%M:%S')

        print_str = f'===     STARTING EXPERIMENT  [{NUM_EXPERIMENTS_RUN}] {exp_name} #{i+1:{c_len}}/{count}  at  {ts_str}     ==='

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
        print_str = f'===     END EXPERIMENT  [{NUM_EXPERIMENTS_RUN}] {exp_name} #{i+1:{c_len}}/{count}  at  {ts_str}  ({e_min}m {e_s:.1f}s)     ==='

        print('='*len(print_str))
        print(print_str)
        print('='*len(print_str))
        print()

    NUM_EXPERIMENTS_RUN += 1


def branch_samples(brnch: PgBranch, ns: List[int]):
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

        for nsamples in branch_samples(brnch, [1, 10]):
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

        for nsamples in branch_samples(branch, [1, 2, 5, 10, 20]):
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

        for nsamples, syncscans in product(branch_samples(branch, [1, 2, 5, 10, 20]), syncscan_ops):
            pgconf = RuntimePgConfig(shared_buffers=shmem,
                                     pbm_evict_num_samples=nsamples,
                                     synchronize_seqscans=syncscans)

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf)


def test_micro_parallelism_same_stream_size(selectivity: float) -> Iterable[ExperimentConfig]:
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')
    base_dbconf = DbConfig(branch=BRANCH_POSTGRES_BASE, sf=10)

    shmem = '2GB'
    cm = 8  # 16 queries per stream
    parallel_ops = [1, 2, 4, 8, 12, 16, 24, 32]
    nsamples = [1, 2, 5, 10, 20]

    for nworkers, branch in product(parallel_ops, POSTGRES_ALL_BRANCHES):
        dbconf = replace(base_dbconf, branch=branch)
        bbconf = BBaseConfig(nworkers=nworkers,
                             workload=WORKLOAD_MICRO_COUNTS.with_multiplier(cm).with_selectivity(selectivity))

        for ns in branch_samples(branch, nsamples):
            pgconf = RuntimePgConfig(shared_buffers=shmem,
                                     pbm_evict_num_samples=ns,
                                     synchronize_seqscans='on')

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf)


def test_WHY_SPIKE(seed: int, parallel_ops=None) -> Iterable[ExperimentConfig]:
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')
    base_dbconf = DbConfig(branch=BRANCH_POSTGRES_BASE, sf=10)

    shmem = '2GB'
    cm = 6  # 12 queries per stream
    if parallel_ops is None:
        # parallel_ops = [1, 2, 4, 6, 8, 12, 16, 24, 32]
        parallel_ops = [2, 4, 8, 12, 16, 24, 32]
    # nsamples = [1, 2, 5, 10, 20]
    nsamples = [1, 5, 10]

    for nworkers, branch in product(parallel_ops, POSTGRES_ALL_BRANCHES):
        dbconf = replace(base_dbconf, branch=branch)
        bbconf = BBaseConfig(nworkers=nworkers, seed=seed,
                             workload=WORKLOAD_MICRO_COUNTS.with_multiplier(cm))

        for ns in branch_samples(branch, nsamples):
            pgconf = RuntimePgConfig(shared_buffers=shmem,
                                     pbm_evict_num_samples=ns,
                                     synchronize_seqscans='on')

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf)


if __name__ == '__main__':
    # Run actual experiments
    # run_tests('test_scripts_buffer_sizes_2', _TEST_test_buffer_sizes())

    # Real tests
    # run_tests('buffer_sizes_3', test_micro_shared_memory())
    # run_tests('parallelism_3', test_micro_parallelism())
    # run_tests('parallelism_same_nqueries_1', test_micro_parallelism_same_stream_size())

    # TODO analyze results of selectivity \/
    # run_tests('parallelism_sel20', test_micro_parallelism_same_stream_size(0.2))
    # run_tests('parallelism_sel40', test_micro_parallelism_same_stream_size(0.4))
    # run_tests('parallelism_sel60', test_micro_parallelism_same_stream_size(0.6))
    # run_tests('parallelism_sel80', test_micro_parallelism_same_stream_size(0.8))

    # TODO try the same thing 6 times, see if it repeats...
    # fill in gaps in the graphs...
    for s in [16312, 22289, 16987, 6262, 32495, 5786]:
        run_tests('test_weird_spike_2', test_WHY_SPIKE(s, [1, 6]))

    # for s in [16312, 22289, 16987, 6262, 32495, 5786]:
    for s in [5786]:
        run_tests('test_weird_spike_2', test_WHY_SPIKE(s))




    ...

#!/usr/bin/env python3
from dataclasses import replace
from typing import Iterable
from datetime import datetime as dt
from itertools import product
import sys

from lib.experiments import *
import lib.config as config


###################
#   HELPER CODE   #
###################

# GLOBALS
NUM_EXPERIMENTS_RUN: int = 0


rand_seeds = [16312, 22289, 16987, 6262, 32495, 5786, 24267, 3636, 9774, 19740, 4448, 19357, 15930, 3127, 4385, 6870, 27272, 14943, 13146, 32540]


def run_tests(exp_name: str, tests: Iterable[ExperimentConfig], /, skip=0, dry_run=False):
    """
    Run a set of experiments.

    skip: skips the first N experiments. So for example if experiment 10 fails, skip should be 9 to re-run experiment 10
    """
    global NUM_EXPERIMENTS_RUN
    tests = list(tests)
    count = len(tests)
    c_len = len(str(count))
    global_start = dt.now()

    if len(tests) == 0:
        return

    for i, exp in enumerate(tests[skip:]):
        start = dt.now()
        ts_str = start.strftime('%H:%M:%S')

        print_str = f'===     STARTING EXPERIMENT  [{NUM_EXPERIMENTS_RUN}] {exp_name} #{i+skip+1:{c_len}}/{count}  at  {ts_str}     ==='

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
        print_str = f'===     END EXPERIMENT  [{NUM_EXPERIMENTS_RUN}] {exp_name} #{i+skip+1:{c_len}}/{count}  at  {ts_str}  ({e_min}m {e_s:.1f}s)     ==='

        print('='*len(print_str))
        print(print_str)
        print('='*len(print_str))
        print()

    global_end = dt.now()
    start_t_str = global_start.strftime('%A, %b %d %H:%M:%S')
    end_t_str = global_end.strftime('%A, %b %d %H:%M:%S')
    elapsed = global_end - global_start
    h = elapsed.seconds // (60*60) + elapsed.days * 24
    m = (elapsed.seconds % (60*60)) // 60
    s = elapsed.seconds % 60
    elapsed_str = f'{h}h {m}m {s}s'
    print_str = f'===     [{exp_name}] STARTED {start_t_str}, FINISHED {end_t_str}, TIME ELAPSED = {elapsed_str}     ==='
    
    print('-'*len(print_str))
    print(print_str)
    print('-'*len(print_str))
    print()

    NUM_EXPERIMENTS_RUN += 1


def branch_samples(brnch: PgBranch, ns: List[int]):
    # Some branches don't support the pbm_num_samples arg at all
    if not brnch.accepts_nsamples:
        return [None]
    # For ones that do, only include case with 1 sample for PBM2\
    # Multiple purely-random tests would be redundant
    elif brnch is BRANCH_PBM2:
        return ns
    else:
        return [n for n in ns if n > 1]


def branch_samples_cache_time(brnch: PgBranch, ns: List[int], ct: List[float]):
    if not brnch.accepts_nsamples:
        return [(None, None)]

    if brnch is BRANCH_PBM2:
        return product(ns, ct)
    else:
        return product([n for n in ns if n > 1], ct)


##############################
#   EXPERIMENT DEFINITIONS   #
##############################


def _TEST_test_script() -> Iterable[ExperimentConfig]:
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')
    bbconf = BBaseConfig(nworkers=2, workload=WORKLOAD_MICRO_WEIGHTS)
    dbdata = DbData(TPCH, sf=2)

    for (shmem, cgmem), brnch in product([('128MB', 1.0), ('128MB', 3.0)], [BRANCH_POSTGRES_BASE]):
        dbbin = DbBin(brnch)
        dbconf = DbConfig(dbbin, dbdata)
        cgroup = CGroupConfig(name=PG_CGROUP, mem_gb=cgmem)

        for nsamples in branch_samples(brnch, [1, 10]):
            pgconf = RuntimePgConfig(shared_buffers=shmem, pbm_evict_num_samples=nsamples)

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf, cgroup=cgroup, db_host='tem114')


def test_micro_shared_memory(seed: int, parallelism=8) -> Iterable[ExperimentConfig]:
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')
    base_bbconf = BBaseConfig(nworkers=parallelism, workload=WORKLOAD_MICRO_COUNTS.with_multiplier(8), seed=seed)
    dbdata = DbData(WORKLOAD_MICRO_COUNTS.workload, sf=10)

    shmem_ops = ['256MB', '512MB', '1GB', '2GB', '4GB', '8GB', '16GB']

    # for shmem, prewarm, branch in product(shmem_ops, [True, False], POSTGRES_ALL_BRANCHES):
    for shmem, prewarm, branch in product(shmem_ops, [True], POSTGRES_ALL_BRANCHES):
        dbbin = DbBin(branch)
        dbconf = DbConfig(dbbin, dbdata)
        bbconf = replace(base_bbconf, prewarm=prewarm)

        for nsamples in branch_samples(branch, [1, 2, 5, 10, 20]):
            pgconf = RuntimePgConfig(shared_buffers=shmem, pbm_evict_num_samples=nsamples)

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf)


def test_micro_parallelism_constant_nqueries() -> Iterable[ExperimentConfig]:
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')
    dbdata = DbData(WORKLOAD_MICRO_COUNTS.workload, sf=10)

    shmem = '2GB'
    total_queries = 2**6
    parallel_ops = [1, 2, 4, 8, 16, 32]
    syncscan_ops = ['on', 'off']
    nsamples = [1, 2, 5, 10, 20]

    for nworkers, branch in product(parallel_ops, POSTGRES_ALL_BRANCHES):
        cm = total_queries // nworkers

        dbbin = DbBin(branch)
        dbconf = DbConfig(dbbin, dbdata)
        bbconf = BBaseConfig(nworkers=nworkers, workload=WORKLOAD_MICRO_COUNTS.with_multiplier(cm))

        for nsamples, syncscans in product(branch_samples(branch, nsamples), syncscan_ops):
            pgconf = RuntimePgConfig(shared_buffers=shmem,
                                     pbm_evict_num_samples=nsamples,
                                     synchronize_seqscans=syncscans)

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf)


def test_tpcc(seeds: List[int], nsamples: List[int] = None, shmem='2560MB', cgmem=3.0,
              blk_sz=DEFAULT_BLOCK_SIZE, bg_sz=DEFAULT_BG_SIZE, use_ssd=False) -> Iterable[ExperimentConfig]:
    ssd_data_root = pathlib.Path('/hdd2/pgdata')
    ssd_dev_stats = 'sda/sda3'
    ssd_host = 'tem06'

    workload = WORKLOAD_TPCC.workload
    data_root = PG_DEFAULT_DATA_ROOT
    if use_ssd:
        workload = workload.with_host_device(ssd_host, ssd_dev_stats)
        data_root = ssd_data_root

    dbdata = DbData(workload, sf=100, block_size=blk_sz, data_root=data_root)
    cgroup = CGroupConfig(cgmem)

    # parallel_ops = [1, 2, 4, 6, 8, 12, 16, 24, 32]
    parallel_ops = [300, 200, 100]
    branches = [BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3]

    nsamples = nsamples or [1, 10]

    for seed, nworkers, branch in product(seeds, parallel_ops, branches):
        dbbin = DbBin(branch, block_size=blk_sz, bg_size=bg_sz)
        dbconf = DbConfig(dbbin, dbdata)

        bbworkload = WORKLOAD_TPCC.with_times(300, 30).with_rate(100)
        bbworkload.workload = workload
        bbconf = BBaseConfig(nworkers=nworkers, seed=seed,
                             workload=bbworkload)
        # TODO or with rate = nworkers? i.e. each worker tries once a second... (maybe have to be less than that)

        for ns in branch_samples(branch, nsamples):
            pgconf = RuntimePgConfig(
                shared_buffers=shmem,
                synchronize_seqscans='on',
                track_io_timing='on',
                pbm_evict_num_samples=ns,
                work_mem='8MB',
                max_connections=nworkers + 5,
                # pbm_evict_num_victims='',
                pbm_bg_naest_max_age=10 if branch.accepts_nsamples else None,
                max_pred_locks_per_transaction=128,  # not sure if needed?
            )

            yield ExperimentConfig(pgconf, dbconf, None, bbconf, cgroup=cgroup, db_host=ssd_host if use_ssd else None)


def test_micro_base(work: CountedWorkloadConfig, seeds: List[Optional[int]], selectivity: float, *,
                    cm=8, parallel_ops: List[int] = None, nsamples: List[int] = None, nvictims: int = 1,
                    cache_time: Optional[float] = None, branches: List[PgBranch] = None,
                    shmem='2GB', cgmem_gb: float = None, blk_sz=DEFAULT_BLOCK_SIZE, bg_sz=DEFAULT_BG_SIZE,
                    data_root: (Path, str) = None, db_host: str = None, extra_pg_args: dict = None) \
        -> Iterable[ExperimentConfig]:
    workload = work.workload
    if data_root is not None:
        workload = workload.with_host_device(db_host, data_root[1])
        dbdata = DbData(workload, sf=10, block_size=blk_sz, data_root=data_root[0])
        # print(f'DbData: {dbdata}')
    else:
        dbdata = DbData(workload, sf=10, block_size=blk_sz)
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')

    cgroup = CGroupConfig(cgmem_gb) if cgmem_gb is not None else None
    if parallel_ops is None:
        parallel_ops = [1, 2, 4, 6, 8, 12, 16, 24, 32]
    if nsamples is None:
        nsamples = [1, 2, 5, 10, 20]
    # TODO make cache_time a list! (or, have a single list of (nsamples, cache time) tuples)
    if branches is None:
        branches = POSTGRES_ALL_BRANCHES

    for seed, nworkers, branch in product(seeds, parallel_ops, branches):
        seed = seed if seed is not None else 12345  # default seed
        dbbin = DbBin(branch, block_size=blk_sz, bg_size=bg_sz)
        dbconf = DbConfig(dbbin, dbdata)
        bbworkload = work.with_multiplier(cm).with_selectivity(selectivity)
        bbworkload.workload = workload
        bbconf = BBaseConfig(nworkers=nworkers, seed=seed, workload=bbworkload)

        for ns in branch_samples(branch, nsamples):
            nv = nvictims if ns is not None and ns > 1 else None
            if ns is not None and ns > 1:
                ns = ns * nv
                nv = nvictims
            pgconf = RuntimePgConfig(shared_buffers=shmem,
                                     pbm_evict_num_samples=ns,
                                     pbm_evict_num_victims=nv,
                                     pbm_bg_naest_max_age=cache_time if branch.accepts_nsamples else None,
                                     synchronize_seqscans='on',
                                     track_io_timing='on',
                                     **(extra_pg_args or {}),
             )

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf, cgroup=cgroup, db_host=db_host)


def test_micro_parallelism(seeds: List[Optional[int]], selectivity: Optional[float], *,
                           cm=8, parallel_ops: List[int] = None, nsamples: List[int] = None, nvictims: int = 1,
                           cache_time: Optional[float] = None, branches: List[PgBranch] = None,
                           shmem='2GB', cgmem_gb: float = None, blk_sz=DEFAULT_BLOCK_SIZE, bg_sz=DEFAULT_BG_SIZE,
                           data_root: (Path, str) = None, db_host: str = None) \
        -> Iterable[ExperimentConfig]:
    return test_micro_base(WORKLOAD_MICRO_COUNTS, seeds, selectivity,
                           cm=cm, parallel_ops=parallel_ops, nsamples=nsamples, nvictims=nvictims,
                           cache_time=cache_time, branches=branches,
                           shmem=shmem, cgmem_gb=cgmem_gb, blk_sz=blk_sz, bg_sz=bg_sz,
                           data_root=data_root, db_host=db_host)


def test_micro_index_parallelism(seeds: List[Optional[int]], selectivity: float, pct_of_range: int, *,
                                 cm=8, parallel_ops: List[int] = None, nsamples: List[int] = None, nvictims: int = 1,
                                 cache_time: Optional[float] = None, branches: List[PgBranch] = None,
                                 shmem='2GB', cgmem_gb: float = None, blk_sz=DEFAULT_BLOCK_SIZE, bg_sz=DEFAULT_BG_SIZE,
                                 data_root: (Path, str) = None, db_host: str = None, extra_pg_args: dict = None) \
        -> Iterable[ExperimentConfig]:

    sel = selectivity + 100 * pct_of_range
    pg_idx_args = {
        'seq_page_cost': 1,
        'random_page_cost': 1.2,
        'enable_bitmapscan': False,
        'enable_seqscan': False,
        **(extra_pg_args or {}),
    }

    return test_micro_base(WORKLOAD_MICRO_IDX_COUNTS, seeds, selectivity=sel,
                           cm=cm, parallel_ops=parallel_ops, nsamples=nsamples, nvictims=nvictims,
                           cache_time=cache_time, branches=branches,
                           shmem=shmem, cgmem_gb=cgmem_gb, blk_sz=blk_sz, bg_sz=bg_sz,
                           data_root=data_root, db_host=db_host, extra_pg_args=pg_idx_args)



def test_micro_parallelism_with_selectivity(selectivity: float) -> Iterable[ExperimentConfig]:
    return test_micro_parallelism([None], selectivity)


def test_micro_parallelism_same_stream_count(seed: int, parallel_ops=None) -> Iterable[ExperimentConfig]:
    return test_micro_parallelism([seed], None, cm=6, parallel_ops=parallel_ops, nsamples=[1, 5, 10])


def test_large_mem(seeds: List[int], blksz=DEFAULT_BLOCK_SIZE, bgsz=DEFAULT_BG_SIZE, *, nvictims=1, sf=100, cgroup=None, shmem='28GB', cm=1) -> Iterable[ExperimentConfig]:
    dbsetup = DbSetup(indexes='lineitem_brinonly', clustering='dates')
    dbdata = DbData(WORKLOAD_MICRO_COUNTS.workload, sf=sf, block_size=blksz)

    # nsamples = [5, 10]
    nsamples = [10]
    # nv = 10
    # nsamples = [1]
    # nsamples = [10, 1]

    # branches = [BRANCH_PBM1, BRANCH_PBM2, BRANCH_POSTGRES_BASE]
    branches = [BRANCH_PBM2]
    # branches = [BRANCH_PBM1, BRANCH_PBM2]
    # branches = [BRANCH_POSTGRES_BASE, BRANCH_PBM1]
    for seed, branch in product(seeds, branches):
        bbconf = BBaseConfig(nworkers=4, seed=seed,
                             workload=WORKLOAD_MICRO_COUNTS.with_multiplier(cm).with_selectivity(0.5))
        dbbin = DbBin(branch, block_size=blksz, bg_size=bgsz)
        dbconf = DbConfig(dbbin, dbdata)
        for ns in branch_samples(branch, nsamples):
            ct = [10.0] if branch.accepts_nsamples else [None]
            if branch.accepts_nsamples:
                nv = nvictims
                ns = ns * nv
            else:
                nv = None
            for t in ct:
                pgconf = RuntimePgConfig(shared_buffers=shmem, pbm_evict_num_samples=ns, pbm_evict_num_victims=nv,
                                         pbm_bg_naest_max_age=t,
                                         track_io_timing='on')

                yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf, cgroup=cgroup)


def rerun_failed(done_count: int, e_str: str, exp: Iterable[ExperimentConfig], dry=True):
    not_tried = list(exp)[done_count:]
    if dry:
        print("First retry:")
        print(not_tried[0].dbconf)
        print(not_tried[0].pgconf)
    else:
        run_tests(e_str, not_tried)


def main_tpch():

    # Run actual experiments
    # run_tests('test_cgroup_1', _TEST_test_script())

    # Real tests
    # run_tests('buffer_sizes_3', test_micro_shared_memory())
    # run_tests('parallelism_3', test_micro_parallelism())
    # run_tests('parallelism_same_nqueries_1', test_micro_parallelism_same_stream_size())

    # run_tests('parallelism_sel20_2', test_micro_parallelism_same_stream_size(0.2))
    # run_tests('parallelism_sel40_2', test_micro_parallelism_same_stream_size(0.4))
    # run_tests('parallelism_sel60_2', test_micro_parallelism_same_stream_size(0.6))
    # run_tests('parallelism_sel80_2', test_micro_parallelism_same_stream_size(0.8))
    #
    for s in [16312, 22289, 16987, 6262, 32495, 5786]:
    #     run_tests('test_weird_spike_3', test_WHY_SPIKE(s))
    #     run_tests('test_pbm3_updated', test_micro_parallelism_same_stream_count(s))
        pass

    # for s in [29020, 29848, 15858]:
    #     run_tests('buffer_sizes_4', test_micro_shared_memory(s))

    # for s in [29020, 29848, 15858]:
    #     run_tests('buffer_sizes_p16_1', test_micro_shared_memory(s, parallelism=16))

    # re-run part which failed...
    # rerun_failed(43, 'buffer_sizes_p16_1', test_micro_shared_memory(15858, parallelism=16))

    # for s in [21473, 25796, 11251, 28834, 16400]:
    #     run_tests('parallelism_sel30_1', test_micro_parallelism(s, 0.3, cm=6, nsamples=[1, 5, 10]))

    # for s in [12345, 23456, 34567]:
    #     run_tests('shmem_at_sf100_with_iostats', test_large_mem(s))
    # `shmem_at_sf100` was before iostats worked, with overflowed timestamps...




    # check whether spinlock vs LWLock makes a difference without any other changes
    # seeds = [16312, 22289, 16987, 6262, 32495, 5786]
    # seeds = [24267, 3636, 9774, 19740, 4448, 19357, 15930, 3127, 4385, 6870, 27272, 14943, 13146, 32540]
    # brnchs = [BRANCH_PBM_COMPARE1, BRANCH_PBM_COMPARE2, BRANCH_PBM2, BRANCH_POSTGRES_BASE, BRANCH_PBM1]
    # brnchs = [BRANCH_POSTGRES_BASE, BRANCH_PBM1]
    # brnchs = [BRANCH_PBM2, BRANCH_PBM_COMPARE1]
    # brnchs = [BRANCH_PBM2]
    # brnchs = [BRANCH_PBM_COMPARE1, BRANCH_PBM_COMPARE2]
    # ename = 'comparing_bg_lock_types'  # pbm2 = BROKEN!, comp1 = no caching + spinlocks, comp2 = no caching + lwlock
    # ename = 'comparing_bg_lock_types_2'  # pbm2 = FIXED caching 1s, comp1 = caching 10s (both spinlocks still)
    # ename = 'comparing_bg_lock_types_3'  # pbm2 = caching 10s & 100s
    # ename = 'comparing_bg_lock_types_4'  # comp1 = caching 10s hard-coded, comp2 = 100s hard-coded
    # for s in seeds:
    #     run_tests(ename, test_micro_parallelism(s, selectivity=0.3, parallel_ops=[32], nsamples=[1], cache_time=None,
    #                                             branches=brnchs))

    # shmem_at_sf100_with_caching_more_iostats
    # shmem_at_sf100_with_caching_more_iostats_bs32
    seeds_sf100 = [12345, 23456, 34567, 5678, 6789]
    # run_tests('shmem_at_sf100_group_eviction_bgsz4096', test_large_mem(seeds_sf100, 8, 4096))
    # run_tests('shmem_at_sf100_group_eviction_bgsz4096', test_large_mem(seeds_sf100, 32, 4096))

    # run_tests('shmem_at_sf100_multi_evict_nv10', test_large_mem(seeds_sf100, 8, 256, nvictims=10))
    # run_tests('shmem_at_sf100_multi_evict_nv10', test_large_mem(seeds_sf100, 32, 256, nvictims=10))
    # run_tests('shmem_at_sf100_multi_evict_nv10', test_large_mem(seeds_sf100, 8, 4096, nvictims=10))
    # run_tests('shmem_at_sf100_multi_evict_nv10', test_large_mem(seeds_sf100, 32, 4096, nvictims=10))
    # run_tests('shmem_at_sf100_multi_evict_nv1', test_large_mem(seeds_sf100, 8, 256, nvictims=1))
    # run_tests('shmem_at_sf100_multi_evict_nv1', test_large_mem(seeds_sf100, 32, 256, nvictims=1))
    # run_tests('shmem_at_sf100_multi_evict_nv1', test_large_mem(seeds_sf100, 8, 4096, nvictims=1))
    # run_tests('shmem_at_sf100_multi_evict_nv1', test_large_mem(seeds_sf100, 32, 4096, nvictims=1))

    # replicate experiments at SF 10
    # run_tests('shmem_multi_evict_sf10_nv10_2', test_large_mem(seeds_sf100, 8, 256, nvictims=10, sf=10, shmem='2560MB', cm=4, cgroup=CGroupConfig(3.0)))
    # run_tests('shmem_multi_evict_sf10_nv10_2', test_large_mem(seeds_sf100, 32, 256, nvictims=10, sf=10, shmem='2560MB', cm=4, cgroup=CGroupConfig(3.0)))
    # run_tests('shmem_multi_evict_sf10_nv10_2', test_large_mem(seeds_sf100, 8, 4096, nvictims=10, sf=10, shmem='2560MB', cm=4, cgroup=CGroupConfig(3.0)))
    # run_tests('shmem_multi_evict_sf10_nv10_2', test_large_mem(seeds_sf100, 32, 4096, nvictims=10, sf=10, shmem='2560MB', cm=4, cgroup=CGroupConfig(3.0)))
    # run_tests('shmem_multi_evict_sf10_nv1_2', test_large_mem(seeds_sf100, 8, 256, nvictims=1, sf=10, shmem='2560MB', cm=4, cgroup=CGroupConfig(3.0)))
    # run_tests('shmem_multi_evict_sf10_nv1_2', test_large_mem(seeds_sf100, 32, 256, nvictims=1, sf=10, shmem='2560MB', cm=4, cgroup=CGroupConfig(3.0)))
    # run_tests('shmem_multi_evict_sf10_nv1_2', test_large_mem(seeds_sf100, 8, 4096, nvictims=1, sf=10, shmem='2560MB', cm=4, cgroup=CGroupConfig(3.0)))
    # run_tests('shmem_multi_evict_sf10_nv1_2', test_large_mem(seeds_sf100, 32, 4096, nvictims=1, sf=10, shmem='2560MB', cm=4, cgroup=CGroupConfig(3.0)))

    # Run SF10 experiments with 3GB system memory, so the OS doesn't just cache everything
    # for s in []:  # seeds[:6]:
    # run_tests('sampling_overhead',
    #           test_micro_parallelism([], selectivity=0.3, cm=4, parallel_ops=[8], nsamples=[1, 10], cache_time=10,
    #                                  cgmem_gb=3.0, branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2],))

    # Similar with 2.5 GiB buffer memomry, 0.5 selectivity, 4 parallel streams to more closely mimic the SF100 experiments
    # run_tests('parallelism_2',
    #           test_micro_parallelism(rand_seeds[0:3], selectivity=0.4, cm=1, parallel_ops=[1, 2, 4, 6, 8, 12, 16, 24, 32],
    #                                  nsamples=[1, 10], cache_time=10, shmem='2560MB', cgmem_gb=3.0,
    #                                  branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3],))
    # TODO ^ shows base == pbm1?? why? :(


    run_tests('parallelism_nocgroup_1',
              test_micro_parallelism(rand_seeds[6:6], selectivity=0.3, cm=6, parallel_ops=[1, 2, 4, 6, 8, 12, 16, 24, 32],
                                     nsamples=[1, 10], cache_time=10, shmem='2560MB', cgmem_gb=None,
                                     branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3],))
    run_tests('parallelism_nocgroup_1',
              test_micro_parallelism(rand_seeds[6:6], selectivity=0.3, cm=6, parallel_ops=[1, 2, 4, 6, 8, 12, 16, 24, 32],
                                     nsamples=[10], cache_time=10, shmem='2560MB', cgmem_gb=None, nvictims=10,
                                     branches=[BRANCH_PBM2,]))  # BRANCH_PBM3],))
    # Done for: 0:6

    run_tests('parallelism_nocgroup_largeblks_1',
              test_micro_parallelism(rand_seeds[6:6], selectivity=0.3, cm=6, parallel_ops=[1, 2, 4, 6, 8, 12, 16, 24, 32],
                                     nsamples=[1, 10], cache_time=10, shmem='2560MB', cgmem_gb=None, blk_sz=32, bg_sz=4096,
                                     branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3],))
    run_tests('parallelism_nocgroup_largeblks_1',
              test_micro_parallelism(rand_seeds[6:6], selectivity=0.3, cm=6, parallel_ops=[1, 2, 4, 6, 8, 12, 16, 24, 32],
                                     nsamples=[10], cache_time=10, shmem='2560MB', cgmem_gb=None, nvictims=10, blk_sz=32, bg_sz=4096,
                                     branches=[BRANCH_PBM2,]))  # BRANCH_PBM3],))
    # Done for: 0:6

    run_tests('parallelism_cgroup_largeblks_1',
              test_micro_parallelism(rand_seeds[3:3], selectivity=0.3, cm=4, parallel_ops=[1, 2, 4, 6, 8, 12, 16, 24, 32],
                                     nsamples=[1, 10], cache_time=10, shmem='2560MB', cgmem_gb=3.0, blk_sz=32, bg_sz=4096,
                                     branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3],))
    run_tests('parallelism_cgroup_largeblks_1',
              test_micro_parallelism(rand_seeds[3:3], selectivity=0.3, cm=4, parallel_ops=[1, 2, 4, 6, 8, 12, 16, 24, 32],
                                     nsamples=[10], cache_time=10, shmem='2560MB', cgmem_gb=3.0, nvictims=10, blk_sz=32, bg_sz=4096,
                                     branches=[BRANCH_PBM2,]))  # BRANCH_PBM3],))
    # Done for: 3:3

    run_tests('parallelism_cgroup_sel50_2',
              test_micro_parallelism(rand_seeds[3:3], selectivity=0.5, cm=4, parallel_ops=[1, 2, 4, 6, 8, 12, 16, 24, 32],
                                     nsamples=[1, 10], cache_time=10, shmem='2560MB', cgmem_gb=3.0, blk_sz=32, bg_sz=4096,
                                     branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3],))
    # Done for: 3:3


    # TODO test with higher selectivity to match the "largemem" tests?


# Run the same test on an ssd on tem06 instead.
def ssd_tests():
    data_root = pathlib.Path('/hdd2/pgdata')
    dev_stats = 'sda/sda3'
    ssd_args = {'data_root': (data_root, dev_stats), 'db_host': 'tem06'}
    common_args = {'parallel_ops': [1, 2, 4, 6, 8, 12, 16, 24, 32], 'cache_time': 10, 'shmem': '2560MB', 'cgmem_gb': 3.0}

    common_seq_args = {'selectivity': 0.3, 'cm': 4, **common_args}
    # run_tests('parallelism_cgroup_largeblks_ssd_2',
    #           test_micro_parallelism(rand_seeds[3:3], **common_args, **ssd_args,
    #                                  nsamples=[1, 10],   branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3], blk_sz=32, bg_sz=4096, ))
    #
    # # repeat with 10 "victims" bulk evictions
    # run_tests('parallelism_cgroup_largeblks_ssd_2',
    #           test_micro_parallelism(rand_seeds[3:3], **common_args, **ssd_args,
    #                                  nsamples=[10],   branches=[BRANCH_PBM2, BRANCH_PBM3], nvictims=10, blk_sz=32, bg_sz=4096, ))
    #
    # # repeat with small blocks
    # run_tests('parallelism_cgroup_smallblks_ssd_2',
    #           test_micro_parallelism(rand_seeds[3:3], **common_args, **ssd_args,
    #                                  nsamples=[1, 10], branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3], ))
    # run_tests('parallelism_cgroup_smallblks_ssd_2',
    #           test_micro_parallelism(rand_seeds[3:3], **common_args, **ssd_args,
    #                                  nsamples=[10], branches=[BRANCH_PBM2, BRANCH_PBM3], nvictims=10, ))

    common_idx_args = {'selectivity': 0.1, 'pct_of_range': 40, 'cm': 8, **common_args}
    # for non-PBM4 as reference
    run_tests('parallelism_idx_ssd_2',
              test_micro_index_parallelism(rand_seeds[3:3], **common_idx_args, **ssd_args, blk_sz=8, bg_sz=1024, nsamples=[1, 10],
                                           branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3],),
              skip=0, dry_run=False)

    # PBM4 NOT using frequency-based stats
    run_tests('parallelism_idx_ssd_pbm4_1',
              test_micro_index_parallelism(rand_seeds[0:3], **common_idx_args, **ssd_args, blk_sz=8, bg_sz=1024, nsamples=[10],
                                           branches=[BRANCH_PBM4], extra_pg_args={'pbm_evict_use_freq': False}),
              skip=0, dry_run=False)
    # TODO consider comparing PBM4 with and without the extra counts
    # TODO also try pbm_evict_use_freq enabled? (technically was enabled for _1 set of tests)



def main_tpcc_hdd():
    run_tests('tpcc_basic_parallelism_3', test_tpcc(rand_seeds[:6], use_ssd=False))
    run_tests('tpcc_basic_parallelism_largeblks_3', test_tpcc(rand_seeds[:6], use_ssd=False, blk_sz=32, bg_sz=4096))


def main_tpcc_ssd():
    run_tests('tpcc_basic_parallelism_ssd_3', test_tpcc(rand_seeds[:6], use_ssd=True, cgmem=4.0))
    run_tests('tpcc_basic_parallelism_largeblks_ssd_3', test_tpcc(rand_seeds[:6], use_ssd=True, cgmem=4.0, blk_sz=32, bg_sz=4096))




if __name__ == '__main__':
    if len(sys.argv) > 1:
        workload_type = sys.argv[1].lower()
    else:
        workload_type = 'tpch'

    # print(f'workload = {workload_type}')

    if workload_type == 'tpch':
        main_tpch()
    elif workload_type == 'tpcc_hdd':
        main_tpcc_hdd()
    elif workload_type == 'tpch_ssd':
        ssd_tests()
    elif workload_type == 'tpcc_ssd':
        main_tpcc_ssd()

    else:
        print(f'unknown workload type "{workload_type}"!')

    # main_tpch()
#!/usr/bin/env python3
from typing import Iterable
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


def test_micro_base(work: CountedWorkloadConfig, seeds: List[Optional[int]], selectivity: float, ssd: bool, *,
                    cm=8, parallel_ops: List[int] = None, nsamples: List[int] = None, nvictims: int = 1,
                    cache_time: Optional[float] = None, branches: List[PgBranch] = None,
                    shmem='2GB', cgmem_gb: float = None, blk_sz=DEFAULT_BLOCK_SIZE, bg_sz=DEFAULT_BG_SIZE,
                    data_root: (Path, str) = None, db_host: str = None,
                    extra_pg_args: dict = None, pbm4_extra_args: dict = None,
                    indexes='lineitem_brinonly', clustering='dates') \
        -> Iterable[ExperimentConfig]:
    workload = work.workload
    if data_root is not None:
        workload = workload.with_host_device(db_host, data_root[1])
        dbdata = DbData(workload, sf=10, block_size=blk_sz, data_root=data_root[0])
        # print(f'DbData: {dbdata}')
    else:
        dbdata = DbData(workload, sf=10, block_size=blk_sz)
    dbsetup = DbSetup(indexes=indexes, clustering=clustering)

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
                                     random_page_cost=1.1 if ssd else None,
                                     **(extra_pg_args or {}),
                                     **(pbm4_extra_args or {} if branch.idx_support else {}),
             )

            yield ExperimentConfig(pgconf, dbconf, dbsetup, bbconf, cgroup=cgroup, db_host=db_host)


def test_micro_parallelism(seeds: List[Optional[int]], selectivity: Optional[float], ssd=True, *,
                           cm=8, parallel_ops: List[int] = None, nsamples: List[int] = None, nvictims: int = 1,
                           cache_time: Optional[float] = None, branches: List[PgBranch] = None,
                           shmem='2GB', cgmem_gb: float = None, blk_sz=DEFAULT_BLOCK_SIZE, bg_sz=DEFAULT_BG_SIZE,
                           data_root: (Path, str) = None, db_host: str = None,
                           indexes='lineitem_brinonly', clustering='dates',
                           pbm4_extra_args: dict = None,) \
        -> Iterable[ExperimentConfig]:
    return test_micro_base(WORKLOAD_MICRO_COUNTS, seeds, selectivity, ssd=ssd,
                           cm=cm, parallel_ops=parallel_ops, nsamples=nsamples, nvictims=nvictims,
                           cache_time=cache_time, branches=branches,
                           shmem=shmem, cgmem_gb=cgmem_gb, blk_sz=blk_sz, bg_sz=bg_sz,
                           data_root=data_root, db_host=db_host,
                           pbm4_extra_args=pbm4_extra_args,
                           indexes=indexes, clustering=clustering,
            )


def test_micro_index_parallelism(seeds: List[Optional[int]], selectivity: float, pct_of_range: int, ssd=True, *,
                                 cm=8, parallel_ops: List[int] = None, nsamples: List[int] = None, nvictims: int = 1,
                                 cache_time: Optional[float] = None, branches: List[PgBranch] = None,
                                 shmem='2GB', cgmem_gb: float = None, blk_sz=DEFAULT_BLOCK_SIZE, bg_sz=DEFAULT_BG_SIZE,
                                 data_root: (Path, str) = None, db_host: str = None,
                                 extra_pg_args: dict = None,  pbm4_extra_args: dict = None,) \
        -> Iterable[ExperimentConfig]:
    """
    micro experiments with index scans

    selectivity: fraction of the table to select in a single scan, from 0 to 1
    pct_of_range: percentage (0-100) of the key-range of the table from which to choose the query bounds
    """

    sel = selectivity + pct_of_range
    pg_idx_args = {
        'enable_bitmapscan': False,
        'enable_seqscan': False,
        **(extra_pg_args or {}),
    }

    return test_micro_base(WORKLOAD_MICRO_IDX_COUNTS, seeds, selectivity=sel, ssd=ssd,
                           cm=cm, parallel_ops=parallel_ops, nsamples=nsamples, nvictims=nvictims,
                           cache_time=cache_time, branches=branches,
                           shmem=shmem, cgmem_gb=cgmem_gb, blk_sz=blk_sz, bg_sz=bg_sz,
                           data_root=data_root, db_host=db_host,
                           extra_pg_args=pg_idx_args, pbm4_extra_args=pbm4_extra_args,
                           indexes='btree')



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


def test_micro_trailing_idx():
    """Experiment: lineitem microbenchmarks with un-correlated index scans, to test "trailing index scan" support."""
    ssd_args = {'data_root': (pathlib.Path('/hdd2/pgdata'), 'sda/sda3'), 'db_host': 'tem06'}
    common_args = {
        'parallel_ops': [1, 2, 4, 6, 8, 12, 16, 24, 32], 'cache_time': 10, 'cgmem_gb': 3.0,
        'selectivity': 0.01, 'pct_of_range': 5, 'cm': 6, 'shmem': '2560MB', 'blk_sz': 8, 'bg_sz': 1024,
        'pbm4_extra_args': {'pbm_evict_use_freq': False, 'pbm_idx_scan_num_counts': 0},
    }

    # for non-PBM4 as reference
    run_tests('micro_idx_parallelism_baseline_1',
              test_micro_index_parallelism(rand_seeds[0:3], **common_args, **ssd_args, nsamples=[1, 10],
                                           branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM2, BRANCH_PBM3],),)

    # PBM4 NOT using frequency-based stats
    run_tests('micro_idx_parallelism_pbm4_1',
              test_micro_index_parallelism(rand_seeds[0:3], **common_args, **ssd_args, nsamples=[10],
                                           branches=[BRANCH_PBM4],),)


def test_micro_seq_index_scans():
    """Experiment: lineitem microbenchmarks with highly correlated index scans to test "sequential index scans" """
    ssd_args = {'data_root': (pathlib.Path('/hdd2/pgdata'), 'sda/sda3'), 'db_host': 'tem06'}
    common_args = {
        **ssd_args,
        'cache_time': 10, 'cgmem_gb': 3.0,
        'selectivity': 0.3, 'cm': 4, 'shmem': '2560MB', 'indexes': 'btree',
        'pbm4_extra_args': {'pbm_evict_use_freq': False},
    }
    standard_args = {'parallel_ops': [1, 2, 4, 6, 8, 12, 16, 24, 32], 'nsamples': [10],}

    # the "fast" ones as baseline...
    run_tests('parallelism_ssd_btree_1', test_micro_parallelism(rand_seeds[3:3], **common_args, **standard_args, branches=[BRANCH_POSTGRES_BASE, BRANCH_PBM1, BRANCH_PBM3,], ))

    # slow baseline configurations... skip 32 parallel workers, we know it is slow
    run_tests('parallelism_ssd_btree_1', test_micro_parallelism(rand_seeds[1:3], **common_args, nsamples=[10], parallel_ops=[1, 2, 4, 6, 8, 12, 16, 24], branches=[BRANCH_PBM2,], ))

    # with support implemented for index scans:
    run_tests('parallelism_ssd_btree_pbm4_1', test_micro_parallelism(rand_seeds[0:3], **common_args, **standard_args, branches=[BRANCH_PBM4,], ))


"""
Main entry point: run the specified experiments in the order given
"""
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise Exception("Expected a workload type!")

    main_experiments = {
        "micro_seq_idx": test_micro_seq_index_scans,
        "micro_trailing_idx": test_micro_trailing_idx,
    }

    # check arguments first
    for main_workload in sys.argv[1:]:
        if main_workload.lower() in main_experiments:
            print(f'Got workload = {main_workload}')
        else:
            raise Exception(f'Unknown experiment: {main_workload}')

    # run the specified workload(s) in order
    for main_workload in sys.argv[1:]:
        test_fn = main_experiments[main_workload.lower()]
        print(f'>>>>> STARTING WORKLOAD = {main_workload} <<<<<')
        test_fn()
        print(f'>>>>> COMPLETED WORKLOAD = {main_workload} <<<<<')

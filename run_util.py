#!/usr/bin/env python3
import argparse

from lib.experiments import *

###################
#  MAIN FUNCTION  #
###################

MAIN_HELP_TEXT = """Action to perform. Actions are:
    pg_setup:           clone and install postgres for each test configuration
    pg_update:          update git repo and rebuild postgres for each test configuration
    pg_clean:           `make clean` for PBM installations
    pg_clean+base:      `make clean` for PBM installations AND the default installation
    bbase_setup:        install benchbase on current host
    bbase_reinstall:    unpack benchbase into the install directory without rebuilding it
    gen_data_tpch:      load test data for the given scale factor for all test configurations
    gen_data_tpcc:      load test data for TPCC, scale factor = # of warehouses
    drop_indexes:       used to remove and indexes and constraints for given scale factor
    reindex:            set the indexes clustering without running benchmarks
    bench:              run benchmarks using the specified scale factor and index type

    testing:            experiments, to be removed...

Note that `bench` runs against postgres installed on a different machine (PG_HOST) and should NOT be run on the postgres
server (it will remotely configure and start/stop postgres as needed) while everything else is setup which runs locally.
(i.e. should be run from the postgres host machine)
"""


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('action', choices=[
        'pg_setup',
        'pg_update',
        'pg_clean',
        'pg_clean+base',
        'bbase_setup',
        'bbase_reinstall',
        'gen_data_tpch',
        'gen_data_tpcc',
        'drop_indexes',
        'reindex',
        'bench',
        'testing'
    ], help=MAIN_HELP_TEXT)
    parser.add_argument('-b', '--branch', type=str, default=None, dest='branch',
                        choices=[b.name for b in POSTGRES_ALL_BRANCHES])
    parser.add_argument('-sf', '--scale-factor', type=int, default=None, dest='sf')
    parser.add_argument('-bs', '--block-size', type=int, default=DEFAULT_BLOCK_SIZE, dest='blk_sz',
                        choices=[1, 2, 4, 8, 16, 32], help='Block size to use (KiB)')
    parser.add_argument('-bgs', '--block-group-size', type=int, default=DEFAULT_BG_SIZE, dest='bg_sz',
                        choices=BLOCK_GROUP_SIZES, help='Size of PBM block groups (KiB)')
    parser.add_argument('-w', '--workload', type=str, default='tpch', choices=[*WORKLOADS_MAP.keys()],
                        help=f'Workload configuration. Options are: {", ".join(WORKLOADS_MAP.keys())}')
    parser.add_argument('-i', '--index-type', type=str, default=None, dest='index_type', metavar='INDEX',
                        help='Folder name under `ddl/index/` with `create.sql` and `drop.sql` to create and drop the '
                             + 'indexes. (e.g. btree)')
    parser.add_argument('-c', '--cluster', type=str, default=None, dest='cluster',
                        help='Script name to cluster tables after indices are created under `ddl/cluster/`. (e.g. pkey)')
    parser.add_argument('-sm', '--shared_buffers', type=str, default='8GB', dest='shmem',
                        help='Amount of memory for PostgreSQL shared buffers. (GB, MB, or kB)')
    parser.add_argument('-p', '--parallelism', type=int, default=8, dest='parallelism',
                        help='Number of terminals (parallel query streams) in BenchBase')
    parser.add_argument('--disable-syncscans', action='store_false', dest='syncscans',
                        help='Disable syncronized scans')
    parser.add_argument('--disable-prewarm', action='store_false', dest='prewarm', help='Disable prewarming')
    parser.add_argument('-e', '--experiment', type=str, default=None, dest='experiment',
                        help='Experiment name to help identify test results')
    parser.add_argument('-cm', '--count-multiplier', type=int, default=4, dest='count_multiplier',
                        help='If using a count workload, the amount to multiply the counts by')
    parser.add_argument('--eviction-samples', type=int, default=None, dest='num_samples',
                        help='Number of eviction samples for sampling-based PBM')
    parser.add_argument('-sel', '--selectivity', type=float, default=None, dest='selectivity',
                        help='Selectivity of the "alt" query types')
    parser.add_argument('--host', type=str, default=None, help='Database hostname (if non-default)')
    args = parser.parse_args()

    if args.action == 'pg_setup':
        one_time_pg_setup()

    elif args.action == 'pg_update':
        refresh_pg_installs()

    elif args.action == 'pg_clean':
        clean_pg_installs(base=False)

    elif args.action == 'pg_clean+base':
        clean_pg_installs(base=True)

    elif args.action == 'bbase_setup':
        one_time_benchbase_setup()

    elif args.action == 'bbase_reinstall':
        install_benchbase()

    elif args.action == 'gen_data_tpch':
        gen_data_tpch(args.sf, blk_sz=args.blk_sz)

    elif args.action == 'gen_data_tpcc':
        gen_data_tpcc(args.sf, blk_sz=args.blk_sz)

    elif args.action == 'drop_indexes':
        drop_all_indexes_tpch(args.sf, blk_sz=args.blk_sz, db_host=args.host)

    elif args.action == 'reindex':
        reindex(args)

    elif args.action == 'bench':
        run_bench(args)

    # TODO remove the 'testing' option
    elif args.action == 'testing':
        pass

        bbconf = BBaseConfig(1, WORKLOAD_MICRO_COUNTS)
        create_bbase_config(1, bbconf, './test_conf_1.xml')

        bbconf = BBaseConfig(1, WORKLOAD_MICRO_COUNTS.with_selectivity(0.3))
        create_bbase_config(1, bbconf, './test_conf_2.xml')

    else:
        raise Exception(f'Unknown action {args.action}')


if __name__ == '__main__':
    main()

"""
Shared configuration between different test scripts.

Some of these should be modified to reproduce the tests on a different set of machines, but most
are constants that should be left alone.
"""

import pathlib
import dataclasses
import typing


###########################
#  PRIMARY CONFIGURATION  #
###########################
# these options should be modified by anyone trying to run this script to reproduce results

# Postgres connection information
PG_HOST_TPCH: str = 'tem112'
PG_HOST_TPCC: str = 'tem114'
PG_PORT: str = '5432'
PG_USER: str = 'ta3vande'
PG_PASSWD: str = ''

# Postgres data files (absolute path)
PG_DATA_ROOT = pathlib.Path('/hdd1/pgdata')
# this is the device on the host which has the given file path.
PG_DATA_DEVICE: str = 'sdb/sdb1'

# Where to clone/compile everything (absolute path)
BUILD_ROOT = pathlib.Path('/home/ta3vande/PG_TESTS')


#######################################
#  CONSTANTS & DERIVED CONFIGURATION  #
#######################################
# These generally should not be modified

@dataclasses.dataclass
class PgBranch:
    gen: int  # "generation" which tracks what features have been implemented
    name: str
    git_branch: str

    @property
    def is_pbm(self) -> bool:
        return self.gen > 0

    @property
    def accepts_nsamples(self) -> bool:
        return self.gen > 1


# Postgres git info: repository and branch names
POSTGRES_GIT_URL = 'ist-git@git.uwaterloo.ca:ta3vande/postgresql-masters-work.git'
# POSTGRES_GIT_URL = 'https://git.uwaterloo.ca/ta3vande/postgresql-masters-work.git'

BRANCH_POSTGRES_BASE = PgBranch(0, 'base', 'REL_14_STABLE')
BRANCH_PBM1 = PgBranch(1, 'pbm1', 'pbm_part1')  # described in paper
BRANCH_PBM2 = PgBranch(2, 'pbm2', 'pbm_part2')  # using sampling-based eviction
BRANCH_PBM3 = PgBranch(2, 'pbm3', 'pbm_part3')  # some support for non-registered buffers
# temporary branches for comparing minor changes:
BRANCH_PBM_OLD = PgBranch(1, 'pbm_old', 'pbm_old')
BRANCH_PBM_COMPARE1 = PgBranch(2, 'pbm_comp1', 'pbm_comp1')
BRANCH_PBM_COMPARE2 = PgBranch(2, 'pbm_comp2', 'pbm_comp2')

POSTGRES_ALL_BRANCHES: typing.List[PgBranch] = [
    BRANCH_POSTGRES_BASE,
    BRANCH_PBM1,
    BRANCH_PBM2,
    BRANCH_PBM3,

    # TEMP: for comparing changes in my own code
    # BRANCH_PBM_OLD,
    # BRANCH_PBM_COMPARE1,
    # BRANCH_PBM_COMPARE2,
]


POSTGRES_PBM_BRANCHES: typing.List[PgBranch] = [
    b for b in POSTGRES_ALL_BRANCHES if b.is_pbm
]

# Derived configuration: paths and branch mappings
POSTGRES_SRC_PATH = BUILD_ROOT / 'pg_src'
POSTGRES_BUILD_PATH = BUILD_ROOT / 'pg_build'
POSTGRES_INSTALL_PATH = BUILD_ROOT / 'pg_install'
POSTGRES_SRC_PATH_BASE = POSTGRES_SRC_PATH / 'base'

# Benchbase
BENCHBASE_GIT_URL = 'ist-git@git.uwaterloo.ca:ta3vande/benchbase.git'
# BENCHBASE_GIT_URL = 'https://@git.uwaterloo.ca/ta3vande/benchbase.git'
BENCHBASE_SRC_PATH = BUILD_ROOT / 'benchbase_src'
BENCHBASE_INSTALL_PATH = BUILD_ROOT / 'benchbase_install'

# Results
RESULTS_ROOT = BUILD_ROOT / 'results'
CONFIG_FILE_NAME = 'test_config.json'
CONSTRAINTS_FILE = 'constraints.csv'
INDEXES_FILE = 'indexes.csv'
IOSTATS_FILE = 'iostats.json'
NON_DIR_RESULTS = [CONFIG_FILE_NAME, CONSTRAINTS_FILE, INDEXES_FILE, IOSTATS_FILE]

# Aggregate results to:
COLLECTED_RESULTS_CSV = 'results.csv'

# Data to remember between runs. These are NOT absolute paths, they are relative to the git repo.
LAST_CONFIG_FILE = 'last_config.json'

# Used to determine the 'pages per range' of BRIN indexes. We want to adjust this depending on the block size to have
# the same number of *rows* per range. (approximately - blocks are padded slightly if not exactly a multiple of the row
# size) This value is divided by the block size (in kB), so it should be a common multiple of all block sizes.
# use 256 KiB so it matches the smallest block group size we're using
BASE_PAGES_PER_RANGE = 32 * 8  # note: 128 is the default 'blocks_per_range' and 8 (kB) is default block size.

# Cgroup to run postgres under to limit total system memory
PG_CGROUP: str = 'postgres_pbm'

# column names for IO stats for the disk
# See https://www.kernel.org/doc/html/latest/block/stat.html for what these columns are
SYSBLOCKSTAT_COLS = [
    'read_ios', 'read_merges', 'sectors_read', 'read_ticks',
    'write_ios', 'write_merges', 'sectors_written', 'write_ticks',
    'in_flight', 'io_ticks', 'time_in_queue',
    'discard_ios', 'discard_merges', 'discard_sectors', 'discard_ticks',
    'flush_ios', 'flush_ticks',
]
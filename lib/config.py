"""
Shared configuration between different test scripts.

Some of these should be modified to reproduce the tests on a different set of machines, but most
are constants that should be left alone.
"""

import pathlib


###########################
#  PRIMARY CONFIGURATION  #
###########################
# these options should be modified by anyone trying to run this script to reproduce results

# Postgres connection information
PG_HOST: str = 'tem112'
PG_PORT: str = '5432'
PG_USER: str = 'ta3vande'
PG_PASSWD: str = ''

# Postgres data files (absolute path)
PG_DATA_ROOT = pathlib.Path('/hdd1/pgdata')

# Where to clone/compile everything (absolute path)
BUILD_ROOT = pathlib.Path('/home/ta3vande/PG_TESTS')


#######################################
#  CONSTANTS & DERIVED CONFIGURATION  #
#######################################
# These generally should not be modified

# Postgres git info: repository and branch names
POSTGRES_GIT_URL = 'ist-git@git.uwaterloo.ca:ta3vande/postgresql-masters-work.git'
# POSTGRES_GIT_URL = 'https://git.uwaterloo.ca/ta3vande/postgresql-masters-work.git'
POSTGRES_BASE_BRANCH = 'REL_14_STABLE'
POSTGRES_PBM_BRANCHES = {
    # key = friendly name in folder paths
    # value = git branch name

    'pbm1': 'pbm_part1',
    'pbm2': 'pbm_part2',

    # TEMP: for comparing changes in my own code...
    # 'pbm_old': 'pbm_old',
}

# Derived configuration: paths and branch mappings
POSTGRES_SRC_PATH = BUILD_ROOT / 'pg_src'
POSTGRES_BUILD_PATH = BUILD_ROOT / 'pg_build'
POSTGRES_INSTALL_PATH = BUILD_ROOT / 'pg_install'
POSTGRES_SRC_PATH_BASE = POSTGRES_SRC_PATH / 'base'
POSTGRES_ALL_BRANCHES = POSTGRES_PBM_BRANCHES.copy()
POSTGRES_ALL_BRANCHES['base'] = POSTGRES_BASE_BRANCH

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
NON_DIR_RESULTS = [CONFIG_FILE_NAME, CONSTRAINTS_FILE, INDEXES_FILE]

# Aggregate results to:
COLLECTED_RESULTS_CSV = 'results.csv'

# Data to remember between runs. These are NOT absolute paths, they are relative to the git repo.
LAST_CONFIG_FILE = 'last_config.json'

# Used to determine the 'pages per range' of BRIN indexes. We want to adjust this depending on the block size to have
# the same number of *rows* per range. (approximately - blocks are padded slightly if not exactly a multiple of the row
# size) This value is divided by the block size (in kB), so it should be a common multiple of all block sizes.
# use 256 KiB so it matches the smallest block group size we're using
BASE_PAGES_PER_RANGE = 32 * 8  # note: 128 is the default 'blocks_per_range' and 8 (kB) is default block size.

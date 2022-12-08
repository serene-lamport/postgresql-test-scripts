#!/usr/bin/env python3
import json
import os
from typing import Optional

import git
import shutil
import subprocess
from datetime import datetime as dt
from pathlib import Path
import postgresql as pg
from git import GitCommandError
from postgresql.api import Connection as PgConnection
from postgresql.installation import Installation, pg_config_dictionary
from postgresql.cluster import Cluster
from postgresql.configfile import ConfigFile
import fabric
from fabric import Connection as FabConnection
import xml.etree.ElementTree as ET
import tqdm
import argparse
from collections import namedtuple
import pandas as pd


###########################
#  PRIMARY CONFIGURATION  #
###########################

# Postgres connection information
PG_HOST: str = 'tem112'
PG_PORT: str = '5432'
PG_USER: str = 'ta3vande'
PG_PASSWD: str = ''

# Postgres data files (absolute path)
PG_DATA_ROOT = Path('/hdd1/pgdata')

# Where to clone/compile everything (absolute path)
BUILD_ROOT = Path('/home/ta3vande/PG_TESTS')


##############################
#  EXPERIMENT CONFIGURATION  #
##############################

# Postgres block size (KiB)
PG_BLK_SIZES = [8, 32]
# PG_BLK_SIZES = [8]

# Time to run tests (s)
# BBASE_TIME = 60
BBASE_TIME = 600

# Allowed sizes of block groups (in KiB). Database is compiled for each of these sizes.
# Must be a power of 2 and multiple of block size
BLOCK_GROUP_SIZES = [256, 1024, 4096]

PG_WORK_MEM = '512MB'


#################################
#  EXTRA/DERIVED CONFIGURATION  #
#################################

# Postgres git info: repository and branch names
POSTGRES_GIT_URL = 'ist-git@git.uwaterloo.ca:ta3vande/postgresql-masters-work.git'
# POSTGRES_GIT_URL = 'https://git.uwaterloo.ca/ta3vande/postgresql-masters-work.git'
POSTGRES_BASE_BRANCH = 'REL_14_STABLE'
POSTGRES_PBM_BRANCHES = {
    # key = friendly name in folder paths
    # value = git branch name
    'pbm1': 'pbm_part1'
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

# Data to remember between runs. These are NOT absolute paths, they are relative to the git repo.
LAST_CONFIG_FILE = 'last_config.json'

# Used to determine the 'pages per range' of BRIN indexes. We want to adjust this depending on the block size to have
# the same number of *rows* per range. (approximately - blocks are padded slightly if not exactly a multiple of the row
# size) This value is divided by the block size (in kB), so it should be a common multiple of all block sizes.
# use 256 KiB so it matches the smallest block group size we're using
BASE_PAGES_PER_RANGE = 32 * 8  # note: 128 is the default 'blocks_per_range' and 8 (kB) is default block size.

##########
#  CODE  #
##########


workload_weights = {
    'tpch': '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0',
    'micro': '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1',
}

# Information about the database is configured: branch/code being used, block size, and scale factor
# These are used to decide which binaries to use (brnch, blk_sz, bg_sz) and which database clster to start (blk_sz, sf)
DbConfig = namedtuple('DbConfig', ['brnch', 'blk_sz', 'bg_sz', 'sf'])
# Information about how to configure benchbase for the test
BBaseConfig = namedtuple('BBaseConfig', ['nworkers', 'results_dir', 'workload'])
# PostgreSQL configuration

# PostgreSQL configuration for the test that isn't relevant for which binary to use (branch and block size) or which
# database cluster (block size and scale factor). These get mapped directly to postgresql.conf so the field names
# should match the config field.
RuntimePgConfig = namedtuple('RuntimePgConfig', ['shared_buffers', 'synchronize_seqscans'])

# Remember how a database is setup: what indexes and clustering?
DbSetup = namedtuple('DbSetup', ['indexes', 'cluster'])


def read_last_config() -> dict:
    try:
        with open(LAST_CONFIG_FILE, 'r') as f:
            decoded = json.JSONDecoder().decode(f.read())
    except FileNotFoundError:
        decoded = []

    last_conf = {}

    for d in decoded:
        d_k = DbConfig(**{k: v for k, v in d.items() if k in DbConfig._fields and k not in ['brnch', 'bg_sz']}, brnch=None, bg_sz=0)
        d_v = DbSetup(**{k: v for k, v in d.items() if k in DbSetup._fields})

        last_conf[d_k] = d_v

    return last_conf


def dbconf_to_dict(c: DbConfig):
    ret = c._asdict()
    del ret['brnch']
    del ret['bg_sz']
    return ret


def update_last_conf(conf: DbConfig, setup: DbSetup):
    last_conf = read_last_config()

    c = conf._replace(brnch=None, bg_sz=0)
    last_conf[c] = setup

    to_encode = [{**dbconf_to_dict(k), **v._asdict()} for k, v in last_conf.items()]

    with open(LAST_CONFIG_FILE, 'w') as f:
        encoder = json.JSONEncoder(indent=2)
        f.write(encoder.encode(to_encode))


class GitProgressBar(git.RemoteProgress):
    """Progress bar for git operations."""
    pbar = None

    def __init__(self, name: str):
        super().__init__()
        self.pbar = tqdm.tqdm(desc=name)

    def update(self, op_code, cur_count, max_count=None, message=""):
        if max_count is not None:
            self.pbar.total = max_count
        self.pbar.update(cur_count - self.pbar.n)

        if message:
            self.pbar.set_postfix(net=message)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.pbar is not None:
            self.pbar.close()


def clone_pg_repos():
    """Clone PostgreSQL repository, including creating worktrees for each postgres branch."""
    with GitProgressBar("PostgreSQL") as pbar:
        pg_repo = git.Repo.clone_from(POSTGRES_GIT_URL, POSTGRES_SRC_PATH_BASE, progress=pbar, multi_options=[f'--branch {POSTGRES_BASE_BRANCH}'])
    print('Creating worktrees for other PostgreSQL branches')
    pbm_repos = []
    for folder, branch in POSTGRES_PBM_BRANCHES.items():
        abs_dir = POSTGRES_SRC_PATH / folder
        pg_repo.git.worktree('add', abs_dir, branch)
        pbm_repos.append(git.Repo(abs_dir))


def clone_benchbase_repo():
    with GitProgressBar("BenchBase ") as pbar:
        bbase_repo = git.Repo.clone_from(BENCHBASE_GIT_URL, BENCHBASE_SRC_PATH, progress=pbar)


def get_repos():
    """Return the already-cloned repositories created by `clone_repos`."""
    pg_repo = git.Repo(POSTGRES_SRC_PATH_BASE)
    pbm_repos = []
    for brnch in POSTGRES_PBM_BRANCHES.keys():
        abs_dir = POSTGRES_SRC_PATH / brnch
        pbm_repos.append(git.Repo(abs_dir))
    bbase_repo = git.Repo(BENCHBASE_SRC_PATH)

    return pg_repo, pbm_repos, bbase_repo


def get_pg_builddir(brnch: str, blk_sz: int, bg_sz: int) -> str:
    if brnch == 'base':
        return f'base_blksz{blk_sz}'
    else:
        return f'{brnch}_blksz{blk_sz}_bgsz{bg_sz}'


def get_build_path(brnch: str, blk_sz: int, bg_sz: int) -> Path:
    return POSTGRES_BUILD_PATH / get_pg_builddir(brnch, blk_sz, bg_sz)


def get_install_path(brnch: str, blk_sz: int, bg_sz: int) -> Path:
    return POSTGRES_INSTALL_PATH / get_pg_builddir(brnch, blk_sz, bg_sz)


def get_data_path(blk_sz: int, tpch_sf: int) -> Path:
    return PG_DATA_ROOT / f'pg_tpch_sf{tpch_sf}_blksz{blk_sz}'


def pbm_blk_shift(blk_sz: int, bg_size: int):
    blks_per_group = bg_size // blk_sz
    res = 0
    while blks_per_group > 1:
        res += 1
        blks_per_group //= 2
    return res


def config_postgres_repo(brnch: str, blk_sz: int, bg_size: int):
    """Runs `./configure` to setup the build for postgres with the provided branch and block size."""
    build_path = get_build_path(brnch, blk_sz, bg_size)
    install_path = get_install_path(brnch, blk_sz, bg_size)
    print(f'Configuring postgres {brnch} with block size {blk_sz}, block group size {bg_size}')
    build_path.mkdir(exist_ok=True, parents=True)

    if brnch == 'base':
        version_str = f'--with-extra-version=-{brnch}_blkzs{blk_sz}'
    else:
        version_str = f'--with-extra-version=-{brnch}_blkzs{blk_sz}_bgsz{bg_size}'

    config_args = [
        POSTGRES_SRC_PATH / brnch / 'configure',
        f'--with-blocksize={blk_sz}',
        f'--prefix={install_path}',
        version_str,
    ]

    # for PBM branches, need some extra config args
    if brnch != 'base':
        bg_shift = pbm_blk_shift(blk_sz, bg_size)
        config_args.append(f'--with-pbmblockshift={bg_shift}')

    subprocess.Popen(config_args, cwd=build_path).wait()


def build_postgres(brnch: str, blk_sz: int, bg_size: int):
    """Compiles PostgreSQL for the specified branch/block size."""
    build_path = get_build_path(brnch, blk_sz, bg_size)
    print(f'Compiling & installing postgres {brnch} with block size {blk_sz} and block group size {bg_size}')
    ret = subprocess.Popen('make', cwd=build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when compiling postgres {brnch} with block size={blk_sz}, group size={bg_size}')
    ret = subprocess.Popen(['make', 'install'], cwd=build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when installing postgres {brnch} with block size={blk_sz}, group size={bg_size}')


def clean_postgres(brnch: str, blk_sz: int, bg_size):
    """Clean PostgreSQL build for the specified branch/block size."""
    build_path = get_build_path(brnch, blk_sz, bg_size)
    print(f'Cleaning postgres {brnch} with block size {blk_sz} and block group size {bg_size}')
    ret = subprocess.Popen(['make', 'clean'], cwd=build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when cleaning postgres {brnch} with block size={blk_sz}, group size={bg_size}')


def build_benchbase():
    """Compile BenchBase."""
    ret = subprocess.Popen([
        BENCHBASE_SRC_PATH / 'mvnw',
        'clean', 'package',
        '-P', 'postgres',
        '-DskipTests'
    ], cwd=BENCHBASE_SRC_PATH).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when compiling benchbase')


def install_benchbase():
    shutil.unpack_archive(BENCHBASE_SRC_PATH / 'target' / 'benchbase-postgres.tgz', BENCHBASE_INSTALL_PATH, 'gztar')


def pg_get_cluster(case: DbConfig) -> Cluster:
    """Return cluster for a local PostgreSQL installation."""
    pgi = pg.installation.Installation(pg_config_dictionary(get_install_path(case.brnch, case.blk_sz, case.bg_sz) / 'bin' / 'pg_config'))
    cl = Cluster(pgi, get_data_path(blk_sz=case.blk_sz, tpch_sf=case.sf))
    return cl


def pg_start_db(cl: Cluster):
    cl.start()
    cl.wait_until_started()


def pg_stop_db(cl: Cluster):
    cl.shutdown()
    cl.wait_until_stopped(timeout=300, delay=0.1)


def pg_init_local_db(cl: Cluster):
    """Initialize a PostgreSQL database cluster, and configure it to accept connections.
    This configures it with essentially no security at all; we assume the host is not accessible
    to the general internet. (in any case, there is nothing on the database except test data)
    """

    # nothing to do if already initialized
    if cl.initialized():
        return

    cl.init()
    with open(cl.hba_file, 'a') as hba:
        hba.writelines(['host\tall\tall\t0.0.0.0/0\ttrust'])
    cl.settings.update({
        'listen_addresses': '*',
        'port': PG_PORT,
        'shared_buffers': '8GB',
        'work_mem': '1GB',
    })


def config_remote_postgres(conn: fabric.Connection, dbconf: DbConfig, pgconf: RuntimePgConfig):
    """Configure a PostgreSQL installation from a different host (the benchmark client).
    This configures it with essentially no security at all; we assume the host is not accessible
    to the general internet. (in any case, there is nothing on the database except test data)
    """
    local_temp_path = 'temp_pg.conf'
    remote_path = str(get_data_path(blk_sz=dbconf.blk_sz, tpch_sf=dbconf.sf) / 'postgresql.conf')

    print(f'Configuring PostgreSQL on remote host, config file at: {remote_path}')

    conn.get(remote_path, local_temp_path)
    cf = ConfigFile(local_temp_path)

    cf.update({
        'listen_addresses': '*',
        'port': PG_PORT,
        'work_mem': PG_WORK_MEM,
        **pgconf._asdict()
    })

    conn.put(local_temp_path, remote_path)
    os.remove(local_temp_path)


def start_remote_postgres(conn: fabric.Connection, case: DbConfig):
    """Start PostgreSQL from the benchmark client machine."""
    install_path = get_install_path(case.brnch, case.blk_sz, case.bg_sz)
    pgctl = install_path / 'bin' / 'pg_ctl'
    data_dir = get_data_path(blk_sz=case.blk_sz, tpch_sf=case.sf)
    logfile = data_dir / 'logfile'

    conn.run(f'truncate --size=0 {logfile}')
    conn.run(f'{pgctl} start -D {data_dir} -l {logfile}')


def stop_remote_postgres(conn: fabric.Connection, case: DbConfig):
    """Stop PostgreSQL remotely."""
    install_path = get_install_path(case.brnch, case.blk_sz, case.bg_sz)
    pgctl = install_path / 'bin' / 'pg_ctl'
    data_dir = get_data_path(blk_sz=case.blk_sz, tpch_sf=case.sf)

    conn.run(f'{pgctl} stop -D {data_dir}')


def create_bbase_config(sf, bb_config: BBaseConfig, out, local=False):
    """Set connection information and scale factor in a BenchBase config file."""
    tree = ET.parse('bbase_config/sample_tpch_config.xml')
    params = tree.getroot()
    params.find('url').text = f'jdbc:postgresql://{"localhost" if local else PG_HOST}:{PG_PORT}/TPCH_{sf}?sslmode=disable&amp;ApplicationName=tpch&amp;reWriteBatchedInserts=true'
    params.find('username').text = PG_USER
    params.find('password').text = PG_PASSWD
    params.find('scalefactor').text = str(sf)
    params.find('terminals').text = str(bb_config.nworkers)

    works = params.find('works')

    # Specify the workload
    for elem in works:
        works.remove(elem)

    work = ET.SubElement(works, 'work')
    ET.SubElement(work, 'serial').text = 'false'
    ET.SubElement(work, 'rate').text = 'unlimited'  # Rate is in queries per second (?)
    ET.SubElement(work, 'weights').text = bb_config.workload
    ET.SubElement(work, 'arrival').text = 'regular'
    ET.SubElement(work, 'time').text = str(BBASE_TIME)  # Time to run the benchmark in seconds (?)
    # ET.SubElement(work, 'warmup').text = '0'

    tree.write(out)


def run_bbase_load(config):
    """Run BenchBase to load data with the given config file path."""
    subprocess.Popen([
        'java',
        '-jar', str(BENCHBASE_INSTALL_PATH / 'benchbase-postgres' / 'benchbase.jar'),
        '-b', 'tpch',
        '-c', str(config),
        '--load=true',
    ], cwd=BENCHBASE_INSTALL_PATH / 'benchbase-postgres').wait()


def run_bbase_test(dbconf: DbConfig, bbconf: BBaseConfig, pgconf: RuntimePgConfig):
    """Run benchbase (on local machine) against PostgreSQL on the remote host.
    Will start & stop PostgreSQL on the remote host.
    """
    temp_bbase_config = BUILD_ROOT / 'bbase_tpch_config.xml'

    create_bbase_config(dbconf.sf, bbconf, temp_bbase_config)

    with FabConnection(PG_HOST) as conn:
        config_remote_postgres(conn, dbconf, pgconf)
        start_remote_postgres(conn, dbconf)

    try:
        subprocess.Popen([
            'java',
            '-jar', str(BENCHBASE_INSTALL_PATH / 'benchbase-postgres' / 'benchbase.jar'),
            '-b', 'tpch',
            '-c', str(temp_bbase_config),
            '--execute=true',
            '-d', str(bbconf.results_dir),
        ], cwd=BENCHBASE_INSTALL_PATH / 'benchbase-postgres').wait()
    finally:
        with FabConnection(PG_HOST) as conn:
            stop_remote_postgres(conn, dbconf)


def pg_exec_file(conn: PgConnection, file):
    with open(file, 'r') as f:
        stmts = ''.join(f.readlines())
    conn.execute(stmts)


def create_and_populate_local_db(case: DbConfig):
    """Initialize a database for the given test case.
    Note: we only need to initialize for the 'base' branch since each branch can use the same data dir
    """
    db_name = f'TPCH_{case.sf}'
    create_ddl_file = 'ddl/create-tables-noindex.sql'
    conn_str = f'pq://localhost/{db_name}'

    cl = pg_get_cluster(case)

    print(f'(Re-)Initializing database cluster at {cl.data_directory}...')
    shutil.rmtree(cl.data_directory, ignore_errors=True)
    pg_init_local_db(cl)

    print(f'Starting cluster and creating database {db_name} with tables (defined in {create_ddl_file}) on local host...')
    pg_start_db(cl)
    try:
        subprocess.run([get_install_path(case.brnch, case.blk_sz, case.bg_sz) / 'bin' / 'createdb', db_name])
        with pg.open(conn_str) as conn:
            conn: PgConnection  # explicitly set type hint since type deduction fails here...
            pg_exec_file(conn, create_ddl_file)
            # Disable autovacuum and WAL for the large tables while loading
            print('Disabling VACUUM on large tables for loading...')
            conn.execute('ALTER TABLE lineitem SET (autovacuum_enabled = off);')
            conn.execute('ALTER TABLE orders SET (autovacuum_enabled = off);')
            conn.execute('ALTER TABLE partsupp SET (autovacuum_enabled = off);')

        print(f'BenchBase: loading test data...')
        bbase_config_file = BUILD_ROOT / 'load_config.xml'
        create_bbase_config(case.sf, BBaseConfig(nworkers=1, results_dir=None, workload=''), bbase_config_file, local=True)
        run_bbase_load(bbase_config_file)

        # Re-enable vacuum after loading is complete
        with pg.open(conn_str) as conn:
            conn: PgConnection  # explicitly set type hint since type deduction fails here...
            # Re-enable autovacuum
            print('Re-enable auto vacuum...')
            conn.execute('ALTER TABLE lineitem SET (autovacuum_enabled = on);')
            conn.execute('ALTER TABLE orders SET (autovacuum_enabled = on);')
            conn.execute('ALTER TABLE partsupp SET (autovacuum_enabled = on);')
            print('ANALYZE large tables...')
            conn.execute('ANALYZE lineitem;')
            conn.execute('ANALYZE orders;')
            conn.execute('ANALYZE partsupp;')

    finally:
        print(f'Shutting down database cluster {cl.data_directory}...')
        pg_stop_db(cl)


def create_indexes(conn: PgConnection, index_dir: str, blk_sz: int):
    if index_dir is None:
        return
    with open(f'ddl/index/{index_dir}/create.sql', 'r') as f:
        lines = f.readlines()
    stmts = ''.join(lines).replace('REPLACEME_BRIN_PAGES_PER_RANGE', str(BASE_PAGES_PER_RANGE / blk_sz))
    conn.execute(stmts)


def drop_indexes(conn: PgConnection, index_dir: str):
    if index_dir is None:
        return
    pg_exec_file(conn, f'ddl/index/{index_dir}/drop.sql')


def cluster_tables(conn: PgConnection, cluster_script: str):
    if cluster_script is None:
        return
    pg_exec_file(conn, f'ddl/cluster/{cluster_script}.sql')


def rename_bbase_results(root: Path):
    """After running benchbase tests, rename the files to remove the date prefix.
    We have a prefix on the folder name instead.
    """

    # os.listdir lists the file names in the directory, not the full path
    pre = None
    f: str
    for f in os.listdir(root):
        i = f.find('.') + 1
        if pre is None:
            pre = f[:i]

        assert f[:i] == pre, 'result files didn\'t all have the same prefix!'

        src = f
        dst = f[i:]

        os.rename(root / src, root / dst)


def reconfigure_indexes(pgconn: PgConnection, blk_sz: int, prev_indexes: str, new_indexes: str):
    # check if the indexes didn't change
    if prev_indexes == new_indexes:
        print(f'Using the same indexes as previously ({prev_indexes}), skipping...')
        return

    # new index type: drop the old ones and create new ones
    print('dropping indexes first if they exist...')
    drop_indexes(pgconn, prev_indexes)

    print(f'create indexes: {new_indexes}')
    create_indexes(pgconn, new_indexes, blk_sz)


def reconfigure_clustering(pgconn: PgConnection, prev_cluster: str, new_cluster: str):
    # check if the clustering didn't change
    if prev_cluster == new_cluster:
        print(f'Using the same clustering as previously ({prev_cluster}), skipping...')
        return

    # re-cluster if using a different method
    print(f'cluster tables: {new_cluster}')
    cluster_tables(pgconn, new_cluster)


def read_constraints_indexes(pgconn: PgConnection):
    # return the constraints and indexes as dataframs
    constraints = pd.DataFrame(pgconn.query("""
        select conrelid::regclass as table, conname as constraint
        from pg_constraint
        where connamespace = 'public'::regnamespace
    """), columns=['table', 'constraint'])

    indexes = pd.DataFrame(pgconn.query("""
        select tablename, indexname, indexdef from pg_indexes
        where schemaname = 'public'
    """), columns=['table', 'index', 'indexdef'])

    return constraints, indexes


def setup_indexes_cluster(blk_sz: int, sf: int, *, prev_indexes: str, new_indexes: str, prev_cluster: str, new_cluster: Optional[str]):
    """Change indexes and clustering on the database. Remembers the changes in `last_config.json`"""

    with FabConnection(PG_HOST) as fabconn:
        dbconf = DbConfig('base', blk_sz=blk_sz, bg_sz=0, sf=sf)
        # Use large amount of memory for creating indexes
        config_remote_postgres(fabconn, dbconf, RuntimePgConfig(shared_buffers='20GB', synchronize_seqscans='on'))
        start_remote_postgres(fabconn, dbconf)

        try:
            with pg.open(f'pq://{PG_HOST}/TPCH_{args.sf}') as pgconn:
                reconfigure_indexes(pgconn, blk_sz, prev_indexes=prev_indexes, new_indexes=new_indexes)
                reconfigure_clustering(pgconn, prev_cluster=prev_cluster, new_cluster=new_cluster or prev_cluster)

                ret = read_constraints_indexes(pgconn)

            # remember the changes
            update_last_conf(dbconf, DbSetup(indexes=new_indexes, cluster=new_cluster))

        finally:
            stop_remote_postgres(fabconn, dbconf)

    return ret


def one_time_pg_setup():
    """Gets postgres installed for each version that is needed.
    Must be run once on the postgres server host
    """

    print('Cloning PostreSQL repo & worktrees...')
    try:
        clone_pg_repos()
    except GitCommandError as e:
        print(f'WARNING: got git error: {e}')
        print(f'    Assuming the repos are already cloned and continuing...')

    # Compile postgres for each different version

    for blk_sz in PG_BLK_SIZES:
        config_postgres_repo('base', blk_sz, 0)
        build_postgres('base', blk_sz, 0)

    for brnch in POSTGRES_PBM_BRANCHES.keys():
        for blk_sz in PG_BLK_SIZES:
            for bg_size in BLOCK_GROUP_SIZES:
                config_postgres_repo(brnch, blk_sz, bg_size)
                build_postgres(brnch, blk_sz, bg_size)


def refresh_pg_installs():
    """Update all git worktrees and rebuild postgress for each configuration.
    Run on the server host.
    """

    # Update each git repo from the remote
    (pg_main_repo, pg_pbm_repos, _) = get_repos()

    with GitProgressBar(f'PostgreSQL {pg_main_repo.active_branch}') as pbar:
        pg_main_repo.remote().pull(progress=pbar)

    for r in pg_pbm_repos:
        with GitProgressBar(f'PostreSQL {r.active_branch}') as pbar:
            r.remote().pull(progress=pbar)

    # Re-compile and re-install postgres for each configuration
    for blk_sz in PG_BLK_SIZES:
        for brnch in POSTGRES_ALL_BRANCHES.keys():
            # touch PBM related files to reduce chance of needing to clean and fully rebuild...
            if brnch != 'base':
                incl_path = POSTGRES_SRC_PATH / brnch / 'src' / 'include' / 'storage'
                (incl_path / 'pbm.h').touch(exist_ok=True)

                src_path = POSTGRES_SRC_PATH / brnch / 'src' / 'backend' / 'storage' / 'buffer'
                (src_path / 'pbm.c').touch(exist_ok=True)
                (src_path / 'pbm_internal.c').touch(exist_ok=True)
                (src_path / 'freelist.c').touch(exist_ok=True)
                (src_path / 'bufmgr.c').touch(exist_ok=True)

            if brnch == 'base':
                build_postgres(brnch, blk_sz, 0)

            else:
                for bg_size in BLOCK_GROUP_SIZES:
                    build_postgres(brnch, blk_sz, bg_size)


def clean_pg_installs(base=False):
    """Clean postgres installations. (only include the base branch if base=True)
    Run on the server host.
    """

    for blk_sz in PG_BLK_SIZES:
        if base:
            clean_postgres('base', blk_sz, 0)

        for brnch in POSTGRES_PBM_BRANCHES.keys():
            for bg_size in BLOCK_GROUP_SIZES:
                clean_postgres(brnch, blk_sz, bg_size)


def one_time_benchbase_setup():
    """Build and install BenchBase on current host."""
    print('Cloning BenchBase...')
    clone_benchbase_repo()
    build_benchbase()
    install_benchbase()


def gen_test_data(sf: int, blk_sz: int):
    if sf is None:
        raise Exception(f'Must specify scale factor when loading data!')

    # Generate test data (only base branch is needed for generating data)
    print('--------------------------------------------------------------------------------')
    print(f'---- Initializing data for blk_sz={blk_sz}, sf={sf}')
    print('--------------------------------------------------------------------------------')
    dbconf = DbConfig('base', blk_sz, bg_sz=0, sf=sf)
    create_and_populate_local_db(dbconf)


def drop_all_indexes(sf: int, blk_sz: int):
    """Drop all indexes and constraints. More powerful cleanup function if something goes really wrong."""
    if sf is None:
        raise Exception(f'Must specify scale factor of databases to clean up!')

    print(f'~~~~~~~~~~ Dropping all indexes and constraints for blk_sz={blk_sz}, sf={sf} ~~~~~~~~~~')
    dbconf = DbConfig('base', blk_sz, bg_sz=0, sf=sf)
    with FabConnection(PG_HOST) as fabconn:
        start_remote_postgres(fabconn, dbconf)

        try:
            with pg.open(f'pq://{PG_HOST}/TPCH_{sf}') as pgconn:
                # find and drop constraints
                all_constraints = pgconn.query("""
                    select conrelid::regclass as table, conname as constraint
                    from pg_constraint
                    where connamespace = 'public'::regnamespace
                """)

                constraints_str = '\n  '.join(f'{c} \t({t})' for t, c in all_constraints)
                print(f'\nFound constraints:\n  {constraints_str}')

                print('Dropping constraints...')
                for t, c in all_constraints:
                    pgconn.execute(f'ALTER TABLE {t} DROP CONSTRAINT IF EXISTS {c} CASCADE')

                # find and drop indexes
                all_indexes = pgconn.query("""
                    select tablename, indexname
                    from pg_indexes
                    where schemaname = 'public'
                """)

                indexes_str = '\n  '.join(f'{idx} \t({t})' for t, idx in all_indexes)
                print(f'\nFound indexes:\n  {indexes_str}')

                print("Dropping indexes...")
                for t, idx in all_indexes:
                    pgconn.execute(f'DROP INDEX IF EXISTS {idx} CASCADE')

                update_last_conf(dbconf, DbSetup(indexes=None, cluster=None))

        finally:
            stop_remote_postgres(fabconn, dbconf)

    print(f'~~~~~~~~~~ All indexes have been dropped ~~~~~~~~~~')


def reindex(args):
    """Change the clustering of the database without running benchmarks."""
    if args.sf is None:
        raise Exception(f'Must specify scale factor of databases to recluster!')

    # read in what indexes/clustering is currently used in the database
    last_config = read_last_config()
    cur_setup: DbSetup = last_config.get(DbConfig(None, args.blk_sz, 0, args.sf)) or DbSetup(None, None)

    prev_indexes = cur_setup.indexes
    prev_cluster = cur_setup.cluster

    # special case: if 'none' is specified we want to forget what the clustering is (and do nothing)
    # in this case set new_cluster to None (handled by setup_indexes_cluster, which updates our state file)
    # if args.cluster is none, it means it was not specified so stay with previous clustering
    forget_clustering = (args.cluster is not None and args.cluster.lower() == 'none')
    if forget_clustering:
        new_cluster = None
    else:
        new_cluster = args.cluster or prev_cluster
    new_indexes = args.index_type or prev_indexes

    print(f'~~~~~~~~~~ Reconfiguring using indexes {new_indexes}, clustering {new_cluster} for blk_sz={args.blk_sz}, sf={args.sf} ~~~~~~~~~~')
    setup_indexes_cluster(args.blk_sz, args.sf,
                          prev_indexes=prev_indexes, new_indexes=new_indexes,
                          prev_cluster=prev_cluster, new_cluster=new_cluster)


def run_benchmarks(args):
    if args.sf is None:
        raise Exception(f'Must specify scale factor!')

    if args.workload not in workload_weights:
        raise Exception(f'Unknown workload type {args.workload}')
    weights = workload_weights[args.workload]

    pgconf = RuntimePgConfig(
        shared_buffers=args.shmem,
        synchronize_seqscans='on' if args.syncscans else 'off',
    )

    # read in what indexes/clustering is currently used in the database
    last_config = read_last_config()
    cur_setup: DbSetup = last_config.get(DbConfig(None, args.blk_sz, 0, args.sf)) or DbSetup(None, None)

    print(f'{cur_setup = }')

    print(f'{last_config = }')

    prev_indexes = cur_setup.indexes
    prev_cluster = cur_setup.cluster

    # create test dir
    ts = dt.now()
    ts_str = ts.strftime('%Y-%m-%d_%H-%M')
    results_dir = RESULTS_ROOT / f'TPCH_{ts_str}'
    os.mkdir(results_dir)

    # store script config in test dir
    test_cluster = args.cluster or prev_cluster
    test_indexes = args.index_type or prev_indexes
    with open(results_dir / CONFIG_FILE_NAME, 'w') as f:
        config = {
            'experiment': args.experiment,
            'clustering': test_cluster,
            'indexes': test_indexes,
            'scalefactor': args.sf,
            'time': BBASE_TIME,
            'parallelism': args.parallelism,
            'workload': args.workload,
            'block_group_size': args.bg_sz,
            'work_mem': PG_WORK_MEM,
            **pgconf._asdict(),
        }
        f.write(json.JSONEncoder(indent=2, sort_keys=True).encode(config))
        f.write('\n')  # ensure trailing newline

    print(f'======================================================================')
    print(f'== Running experiments with:')
    print(f'==   Time:                  {BBASE_TIME // 60} min{"" if BBASE_TIME % 60 == 0 else str(BBASE_TIME % 60) + " s"}')
    print(f'==   Scale factor:          {args.sf}')
    print(f'==   Block size:            {args.blk_sz} KiB')
    print(f'==   Block group size       {args.bg_sz} KiB')
    print(f'==   Workload:              {args.workload}     (weights: {weights})')
    print(f'==   Shared memory:         {args.shmem}')
    print(f'==   Worker memory          {PG_WORK_MEM}')
    print(f'==   Index definitions:     ddl/index/{test_indexes}/   (args.index_type)')
    print(f'==   Clustering:            dd/cluster/{test_cluster}.sql  ({args.cluster})')
    print(f'==   Terminals:             {args.parallelism}')
    print(f'==   SyncScans:             {pgconf.synchronize_seqscans}')
    print(f'== Storing results to {results_dir}')
    print(f'======================================================================')

    # Create indexes once for the block size before the tests with different branches
    print(f'~~~~~~~~~~ Setup indexes={test_indexes}, clustering={test_cluster} for blk_sz={args.blk_sz}, sf={args.sf} ~~~~~~~~~~')

    constraints, indexes = setup_indexes_cluster(args.blk_sz, args.sf,
                                                 prev_indexes=prev_indexes, new_indexes=test_indexes,
                                                 prev_cluster=prev_cluster, new_cluster=test_cluster)



    with open(results_dir / CONSTRAINTS_FILE, 'w') as f:
        constraints.to_csv(f, index=False)

    with open(results_dir / INDEXES_FILE, 'w') as f:
        indexes.to_csv(f, index=False)




    print(f'~~~~~~~~~~ Index and clustering setup done! Running the real tests... ~~~~~~~~~~')

    # Actually run the tests
    for brnch in POSTGRES_ALL_BRANCHES.keys():
        print('--------------------------------------------------------------------------------')
        print(f'---- Running experiments for branch={brnch}, blk_sz={args.blk_sz}, bg_size={args.bg_sz}, sf={args.sf}')
        print('--------------------------------------------------------------------------------')

        dbconf = DbConfig(brnch, args.blk_sz, args.bg_sz, sf=args.sf)
        results_subdir = results_dir / f'{brnch}_blksz{args.blk_sz}'

        run_bbase_test(dbconf, BBaseConfig(nworkers=args.parallelism, results_dir=results_subdir, workload=weights), pgconf)
        rename_bbase_results(results_subdir)


###################
#  MAIN FUNCTION  #
###################

MAIN_HELP_TEXT = """Action to perform. Actions are:
    pg_setup:           clone and install postgres for each test configuration
    pg_update:          update git repo and rebuild postgres for each test configuration
    pg_clean:           `make clean` for PBM installations
    pg_clean+base:      `make clean` for PBM installations AND the default installation
    benchbase_setup:    install benchbase on current host
    gen_test_data:      load test data for the given scale factor for all test configurations
    drop_indexes:       used to remove and indexes and constraints for given scale factor
    reindex:            set the indexes clustering without running benchmarks
    bench:              run benchmarks using the specified scale factor and index type

    testing:            experiments, to be removed...

Note that `bench` runs against postgres installed on a different machine (PG_HOST) and should NOT be run on the postgres
server (it will remotely configure and start/stop postgres as needed) while everything else is setup which runs locally.
(i.e. should be run from the postgres host machine)
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('action', choices=[
        'pg_setup',
        'pg_update',
        'pg_clean',
        'pg_clean+base',
        'benchbase_setup',
        'gen_test_data',
        'drop_indexes',
        'reindex',
        'bench',
        'testing'
    ], help=MAIN_HELP_TEXT)
    parser.add_argument('-sf', '--scale-factor', type=int, default=None, dest='sf')
    parser.add_argument('-bs', '--block-size', type=int, default=8, dest='blk_sz', choices=[1,2,4,8,16,32],
                        help='Block size to use (KiB)')
    parser.add_argument('-bgs', '--block-group-size', type=int, default=256, dest='bg_sz', choices=BLOCK_GROUP_SIZES,
                        help='Size of PBM block groups (KiB)')
    parser.add_argument('-w', '--workload', type=str, default='tpch', choices=[*workload_weights.keys()],
                        help=f'Workload configuration. Options are: {", ".join(workload_weights.keys())}')
    parser.add_argument('-i', '--index-type', type=str, default=None, dest='index_type', metavar='INDEX',
                        help='Folder name under `ddl/index/` with `create.sql` and `drop.sql` to create and drop the indexes. (e.g. btree)')
    parser.add_argument('-c', '--cluster', type=str, default=None, dest='cluster',
                        help='Script name to cluster tables after indices are created under `ddl/cluster/`. (e.g. pkey)')
    parser.add_argument('-sm', '--shared_buffers', type=str, default='8GB', dest='shmem',
                        help='Amount of memory for PostgreSQL shared buffers. (GB, MB, or kB)')
    parser.add_argument('-p', '--parallelism', type=int, default=8, dest='parallelism',
                        help='Number of terminals (parallel query streams) in BenchBase')
    parser.add_argument('--disable-syncscans', action='store_false', dest='syncscans',
                        help='Disable syncronized scans')
    parser.add_argument('-e', '--experiment', type=str, default=None, dest='experiment',
                        help='Experiment name to help identify test results')
    args = parser.parse_args()

    if args.action == 'pg_setup':
        one_time_pg_setup()

    elif args.action == 'pg_update':
        refresh_pg_installs()

    elif args.action == 'pg_clean':
        clean_pg_installs(base=False)

    elif args.action == 'pg_clean+base':
        clean_pg_installs(base=True)

    elif args.action == 'benchbase_setup':
        one_time_benchbase_setup()

    elif args.action == 'gen_test_data':
        gen_test_data(args.sf, blk_sz=args.blk_sz)

    elif args.action == 'drop_indexes':
        drop_all_indexes(args.sf, blk_sz=args.blk_sz)

    elif args.action == 'reindex':
        reindex(args)

    elif args.action == 'bench':
        run_benchmarks(args)

    # TODO remove the 'testing' option
    elif args.action == 'testing':
        pass

        import pandas as pd

        with FabConnection(PG_HOST) as fabconn:

            dbconf = DbConfig('base', 8, 256, 1)

            try:
                start_remote_postgres(fabconn, dbconf)

                with pg.open(f'pq://{PG_HOST}/TPCH_1') as pgconn:
                    pgconn: PgConnection
                    constraints, indexes = read_constraints_indexes(pgconn)

            finally:
                stop_remote_postgres(fabconn, dbconf)


    else:
        raise Exception(f'Unknown action {args.action}')






# TODO: ...
# - [?] re-build with consistent block group size (configure shift amounts)
# - [ ] more build configs with different block-group sizes, other config variables? (or branches?)
# - [ ] make block size and other compile-time variables arguments? so one call only tests each branch (2 for now, instead of 4+ tests, increasing multiplicatively...)
#   - [ ] this requires making the last_cluster/last_indexes more sophisticated... need to remember what it is for every configuration!
# - [ ] figure out file locks thing!
# - [ ] ...

# MAYBE: ...
# - [ ] make generating data allowed from remote host so benchbase is ony needed on one machine

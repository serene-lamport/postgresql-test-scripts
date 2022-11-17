#!/usr/bin/env python3
import os
import git
import shutil
import subprocess
from pathlib import Path
import postgresql as pg
import postgresql.api
from postgresql.installation import Installation, pg_config_dictionary
from postgresql.cluster import Cluster
from postgresql.configfile import ConfigFile
import fabric
# from fabric import Connection
import xml.etree.ElementTree as ET
import tqdm

from collections import namedtuple


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
ROOT_DIR = Path('/home/ta3vande/PG_TESTS')


##############################
#  EXPERIMENT CONFIGURATION  #
##############################

# Postgres block size (KiB)
PG_BLK_SIZES = [8, 32]


################################
#  EXTRA/DERIVD CONFIGURATION  #
################################

# Postgres git info: repository and branch names
POSTGRES_GIT_URL = 'ist-git@git.uwaterloo.ca:ta3vande/postgresql-masters-work.git'
POSTGRES_BASE_BRANCH = 'REL_14_STABLE'
POSTGRES_PBM_BRANCHES = {
    # key = friendly name in folder paths
    # value = git branch name
    'pbm1': 'pbm_part1'
}

# Derived configuration: paths and branch mappings
POSTGRES_SRC_PATH = ROOT_DIR / 'pg_src'
POSTGRES_BUILD_PATH = ROOT_DIR / 'pg_build'
POSTGRES_INSTALL_PATH = ROOT_DIR / 'pg_install'
POSTGRES_SRC_PATH_BASE = POSTGRES_SRC_PATH / 'base'
POSTGRES_ALL_BRANCHES = POSTGRES_PBM_BRANCHES.copy()
POSTGRES_ALL_BRANCHES['base'] = POSTGRES_BASE_BRANCH

# Benchbase
BENCHBASE_GIT_URL = 'https://github.com/cmu-db/benchbase.git'
BENCHBASE_SRC_PATH = ROOT_DIR / 'benchbase_src'
BENCHBASE_INSTALL_PATH = ROOT_DIR / 'benchbase_install'


##########
#  CODE  #
##########

# TODO other config options: ammount of shared memory?

# Information about the database is configured: branch/code being used, block size, and scale factor
DbConfig = namedtuple('DbConfig', ['brnch', 'blk_sz', 'sf'])
# Information about how the test is configured: parallelism, etc... TODO what else?
TestConfig = namedtuple('TestConfig', ['nworkers'])

# TODO other config parameters


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


def clone_repos():
    """Clone PostgreSQL and BenchBase repositories, including creating worktrees for each postgres branch."""
    print('Cloning repositories')

    with GitProgressBar("PostgreSQL") as pbar:
        pg_repo = git.Repo.clone_from(POSTGRES_GIT_URL, POSTGRES_SRC_PATH_BASE, progress=pbar, multi_options=[f'--branch {POSTGRES_BASE_BRANCH}'])
    print('Creating worktrees for other PostgreSQL branches')
    pbm_repos = []
    for folder, branch in POSTGRES_PBM_BRANCHES.items():
        abs_dir = POSTGRES_SRC_PATH / folder
        pg_repo.git.worktree('add', abs_dir, branch)
        pbm_repos.append(git.Repo(abs_dir))

    with GitProgressBar("BenchBase ") as pbar:
        bbase_repo = git.Repo.clone_from(BENCHBASE_GIT_URL, BENCHBASE_SRC_PATH, progress=pbar)

    return pg_repo, pbm_repos, bbase_repo


def get_repos():
    """Return the already-cloned repositories created by `clone_repos`."""
    pg_repo = git.Repo(POSTGRES_SRC_PATH_BASE)
    pbm_repos = []
    for brnch in POSTGRES_PBM_BRANCHES.keys():
        abs_dir = POSTGRES_SRC_PATH / brnch
        pbm_repos.append(git.Repo(abs_dir))
    bbase_repo = git.Repo(BENCHBASE_SRC_PATH)

    return pg_repo, pbm_repos, bbase_repo


def get_build_path(brnch: str, blk_sz: int) -> Path:
    return POSTGRES_BUILD_PATH / f'{brnch}_{blk_sz}'


def get_install_path(brnch: str, blk_sz: int) -> Path:
    return POSTGRES_INSTALL_PATH / f'{brnch}_{blk_sz}'


def get_data_path(blk_sz: int, tpch_sf) -> Path:
    return PG_DATA_ROOT / f'pg_tpch_sf{tpch_sf}_blksz{blk_sz}'


def config_postgres(brnch: str, blk_sz: int):
    """Runs `./configure` to setup the build for postgres with the provided branch and block size."""
    build_path = get_build_path(brnch, blk_sz)
    install_path = get_install_path(brnch, blk_sz)
    print(f'Configuring postgres {brnch} with block size {blk_sz}')
    build_path.mkdir(exist_ok=True, parents=True)
    subprocess.Popen([
        POSTGRES_SRC_PATH / brnch / 'configure',
        f'--with-blocksize={blk_sz}',
        f'--prefix={install_path}',
        f'--with-extra-version={brnch}_{blk_sz}',
    ], cwd=build_path).wait()


def build_postgres(brnch: str, blk_sz: int):
    """Compiles PostgreSQL for the specified branch/block size."""
    build_path = get_build_path(brnch, blk_sz)
    print(f'Compiling & installing postgres {brnch} with block size {blk_sz}')
    ret = subprocess.Popen('make', cwd=build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when compiling postgres {brnch} with block size={blk_sz}')
    ret = subprocess.Popen(['make', 'install'], cwd=build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when installing postgres {brnch} with block size={blk_sz}')


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
    pgi = pg.installation.Installation(pg_config_dictionary(get_install_path(case.brnch, case.blk_sz) / 'bin' / 'pg_config'))
    cl = Cluster(pgi, get_data_path(blk_sz=case.blk_sz, tpch_sf=case.sf))
    return cl


def pg_start_db(cl: Cluster):
    cl.start()
    cl.wait_until_started()


def pg_stop_db(cl: Cluster):
    cl.shutdown()
    cl.wait_until_stopped()


def pg_init_db(cl: Cluster):
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
    })


def config_remote_postgres(conn: fabric.Connection, blk_sz: int, sf):
    """Configure a PostgreSQL installation from a different host (the benchmark client).
    This configures it with essentially no security at all; we assume the host is not accessible
    to the general internet. (in any case, there is nothing on the database except test data)
    """
    local_temp_path = 'temp_pg.conf'
    remote_path = str(get_data_path(blk_sz=blk_sz, tpch_sf=sf) / 'postgresql.conf')

    print(f'Configuring PostgreSQL on remote host, config file at: {remote_path}')

    conn.get(remote_path, local_temp_path)
    cf = ConfigFile(local_temp_path)

    cf.update({
        'listen_addresses': '*',
        'port': PG_PORT,
        'shared_buffers': '8GB',  # TODO how much shared memory? make this a configuration parameter?
    })

    conn.put(local_temp_path, remote_path)
    os.remove(local_temp_path)


def start_remote_postgres(conn: fabric.Connection, case: DbConfig):
    """Start PostgreSQL from the benchmark client machine."""
    install_path = get_install_path(case.brnch, case.blk_sz)
    pgctl = install_path / 'bin' / 'pg_ctl'
    data_dir = get_data_path(blk_sz=case.blk_sz, tpch_sf=case.sf)
    logfile = data_dir / 'logfile'

    conn.run(f'truncate --size=0 {logfile}')
    conn.run(f'{pgctl} start -D {data_dir} -l {logfile}')


def stop_remote_postgres(conn: fabric.Connection, case: DbConfig):
    """Stop PostgreSQL remotely."""
    install_path = get_install_path(case.brnch, case.blk_sz)
    pgctl = install_path / 'bin' / 'pg_ctl'
    data_dir = get_data_path(blk_sz=case.blk_sz, tpch_sf=case.sf)

    conn.run(f'{pgctl} stop -D {data_dir}')


def create_bbase_config(sf, bb_config: TestConfig, out):
    """Set connection information and scale factor in a BenchBase config file."""
    tree = ET.parse('bbase_config/sample_tpch_config.xml')
    params = tree.getroot()
    params.find('url').text = f'jdbc:postgresql://{PG_HOST}:{PG_PORT}/TPCH_{sf}?sslmode=disable&amp;ApplicationName=tpch&amp;reWriteBatchedInserts=true'
    params.find('username').text = PG_USER
    params.find('password').text = PG_PASSWD
    params.find('scalefactor').text = str(sf)
    params.find('terminals').text = str(bb_config.nworkers)

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


def run_bbase_test(conn: fabric.Connection, case: DbConfig, bb_config: TestConfig):
    """Run benchbase (on local machine) against PostgreSQL on the remote host.
    Will start & stop PostgreSQL on the remote host.
    """
    temp_bbase_config = ROOT_DIR / 'bbase_tpch_config.xml'

    create_bbase_config(case.sf, bb_config, temp_bbase_config)
    config_remote_postgres(conn, case.blk_sz, case.sf)
    start_remote_postgres(conn, case)
    # TODO: maybe better to close the connection while running the test?

    subprocess.Popen([
        'java',
        '-jar', str(BENCHBASE_INSTALL_PATH / 'benchbase-postgres' / 'benchbase.jar'),
        '-b', 'tpch',
        '-c', str(temp_bbase_config),
        '--execute=true',
    ], cwd=BENCHBASE_INSTALL_PATH / 'benchbase-postgres').wait()

    stop_remote_postgres(conn, case)


def pg_exec_file(conn: postgresql.api.Connection, file):
    with open(file, 'r') as f:
        stmts = ''.join(f.readlines())
    conn.execute(stmts)


def create_and_populate_db(cl: Cluster, case: DbConfig):
    """Initialize a database for the given test case.
    Note: we only need to initialize for the 'base' branch since
    """
    pg_init_db(cl)
    pg_start_db(cl)
    subprocess.run([get_install_path(case.brnch, case.blk_sz) / 'bin' / 'createdb', f'TPCH_{case.sf}'])
    conn = pg.open(f'pq://{PG_HOST}/TPCH_{case.sf}')
    # with open('ddl/postgres-noindex.sql', 'r') as f:
    #     create_ddl = ''.join(f.readlines())
    # conn.execute(create_ddl)
    pg_exec_file(conn, 'ddl/create-tables-noindex.sql')

    create_bbase_config(sf, TestConfig(nworkers=5), ROOT_DIR / 'load_config.xml')
    run_bbase_load(ROOT_DIR / 'load_config.xml')
    pg_stop_db(cl)




if __name__ == '__main__':
    # (pg_repo, pbm_repos, bbase_repo) = clone_repos()

    # for brnch in POSTGRES_ALL_BRANCHES.keys():
    #     for blk_sz in PG_BLK_SIZES:
    #         config_postgres(brnch, blk_sz)
    #         build_postgres(brnch, blk_sz)

    # build_benchbase()
    # install_benchbase()

    sf = 1
    case = DbConfig(brnch='base', blk_sz=8, sf=sf)

    cl = pg_get_cluster(case)
    create_and_populate_db(cl, case)

    # conn = fabric.Connection(PG_HOST)
    # run_bbase_test(conn, case)



# TODO: ...

# Setup on PG server:
# - [x] clone PG and create worktrees
# - [x] `configure` and `make`, `make install` for each worktree/branch AND eack block size!
#    - [x] also set the actual configuration
# - [x] for each block size (but not branch!), create a DB cluster and database

# Setup on benchbase server:
# - [x] clone, compile, and decompress benchbase (or just decompress, if using pre-built)

# Loading data:
# - [x] need to start postgres. Need to make sure we get the right one!
# - [ ] can run benchbase from any server to load the data: generate config file for desired scale factor and "--load=true"
# - [ ] stop postgres again!

# Running tests:
# - ideally: run only on the benchbase node...
# - [x] configure postgres (remotely! might be tricky...)
# - [ ] generate configuration for benchbase (don't assume we already have it)
# - [ ] ...



# ALSO NEED TO DO!
# - [ ] decide what indices to use, how to cluster tables
# - [ ] decide what workloads to test
# - [ ] microbenchmarks?
# - [ ] sort out indices!
# - [ ] ...

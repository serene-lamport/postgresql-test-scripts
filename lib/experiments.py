"""
Code for setting up and running experiments.
"""
import json
import os
from typing import Optional, List, Dict, DefaultDict, Union
from abc import ABC, abstractmethod
import copy

import git
import shutil
import subprocess
from datetime import datetime as dt
import time
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
from collections import defaultdict
from dataclasses import dataclass, asdict, fields, field
import pandas as pd

from lib.config import *


##############################
#  EXPERIMENT CONFIGURATION  #
##############################

# Postgres block size (KiB)
# PG_BLK_SIZES = [8, 32]
PG_BLK_SIZES = [8]

# Time to run tests (s)
# BBASE_TIME = 600
BBASE_TIME = 200
BBASE_WARMUP_TIME = 10

# Allowed sizes of block groups (in KiB). Database is compiled for each of these sizes.
# Must be a power of 2 and multiple of block size
# BLOCK_GROUP_SIZES = [256, 1024, 4096]
BLOCK_GROUP_SIZES = [256]

PG_WORK_MEM = '32MB'

# Defaults for parameters with multiple options
DEFAULT_BLOCK_SIZE = 8
DEFAULT_BG_SIZE = 256


##########
#  CODE  #
##########


@dataclass(frozen=True)
class Workload:
    name: str
    base_config_file: str
    default_db_host: str
    device: str

    def db_name(self, sf: int):
        return f'{self.name.upper()}_{sf}'


@dataclass
class WorkloadConfig(ABC):
    """Abstract class representing a benchbase workload. This knows how to configure benchbase."""
    workname: str
    workload: Workload
    selectivity: Optional[float] = field(init=False, default=None)

    def with_selectivity(self, selectivity: Optional[float]):
        """Set the selectivity for the 'alt' queries."""
        ret = copy.copy(self)
        ret.selectivity = selectivity
        return ret

    @abstractmethod
    def write_bbase_config(self, work_element: ET.Element): ...

    @abstractmethod
    def to_config_map(self) -> dict: ...


@dataclass
class WeightedWorkloadConfig(WorkloadConfig):
    """Benchbase workload where relative frequencies are specified for query types, and a total time limit is given"""
    weights: str
    time_s: int

    warmup_s: int = 0
    rate: str = 'unlimited'
    arrival: str = 'regular'

    def with_rate(self, q_per_s: int) -> 'WeightedWorkloadConfig':
        ret = copy.copy(self)
        ret.rate = str(q_per_s)
        return ret

    def with_poisson_arrival(self) -> 'WeightedWorkloadConfig':
        ret = copy.copy(self)
        ret.arrival = 'poisson'
        return ret

    def with_times(self, time_s, warmup) -> 'WeightedWorkloadConfig':
        ret = copy.copy(self)
        ret.time_s = time_s
        ret.warmup_s = warmup
        return ret

    def write_bbase_config(self, work_element: ET.Element):
        ET.SubElement(work_element, 'rate').text = self.rate
        ET.SubElement(work_element, 'arrival').text = self.arrival
        ET.SubElement(work_element, 'warmup').text = str(self.warmup_s)
        ET.SubElement(work_element, 'weights').text = self.weights
        ET.SubElement(work_element, 'time').text = str(self.time_s)

    def to_config_map(self) -> dict:
        return {
            # general workload config fields
            'workload': self.workname,
            'selectivity': self.selectivity if self.selectivity is not None else '',
            # specific to this workload type:
            'time': self.time_s,
            'warmup': self.warmup_s,
            'rate': self.rate,
            'arrival': self.arrival,
        }


@dataclass
class CountedWorkloadConfig(WorkloadConfig):
    """Benchbase workload where each query is run a certain number of times (from each worker)"""
    counts: List[int]
    count_multiplier: int = 1

    def with_multiplier(self, cm) -> 'CountedWorkloadConfig':
        """Set the count for each query in each thread"""
        ret = copy.copy(self)
        ret.count_multiplier = cm

        return ret

    def write_bbase_config(self, work_element: ET.Element):
        ET.SubElement(work_element, 'rate').text = 'unlimited'
        ET.SubElement(work_element, 'arrival').text = 'regular'
        # ET.SubElement(work_element, 'warmup').text = '0'
        ET.SubElement(work_element, 'counts').text = ','.join(str(c * self.count_multiplier) for c in self.counts)

    def to_config_map(self) -> dict:
        return {
            # general workload config fields
            'workload': self.workname,
            'selectivity': self.selectivity if self.selectivity is not None else '',
            # specific to this workload type:
            'count_multiplier': self.count_multiplier,
        }


# The available workloads
TPCH = Workload('tpch', 'bbase_config/sample_tpch_config.xml', PG_HOST_TPCH, PG_DATA_DEVICE)
TPCC = Workload('tpcc', 'bbase_config/sample_tpcc_config.xml', PG_HOST_TPCC, PG_DATA_DEVICE)


WORKLOAD_TPCH_WEIGHTS = WeightedWorkloadConfig('tpch_w', TPCH, weights='1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,0', time_s=BBASE_TIME)
WORKLOAD_MICRO_WEIGHTS = WeightedWorkloadConfig('micro_w', TPCH, weights='0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1', time_s=BBASE_TIME)
WORKLOAD_TPCH_COUNTS = CountedWorkloadConfig('tpch_c', TPCH, counts=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0])
WORKLOAD_MICRO_COUNTS = CountedWorkloadConfig('micro_c', TPCH, counts=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
WORKLOAD_TPCC = WeightedWorkloadConfig('tpcc', TPCC, weights='45,43,4,4,4', time_s=BBASE_TIME, warmup_s=BBASE_WARMUP_TIME)


WORKLOADS_MAP: Dict[str, WorkloadConfig] = {c.workname: c for c in [
    WORKLOAD_TPCH_WEIGHTS,
    WORKLOAD_MICRO_WEIGHTS,
    WORKLOAD_TPCH_COUNTS,
    WORKLOAD_MICRO_COUNTS,
    WORKLOAD_TPCC,
]}


@dataclass(frozen=True, eq=True)
class ConfigKey:
    """Key in the `last_config.json` file"""
    sf: int
    blk_sz: int


@dataclass(frozen=True)
class DbBin:
    """
    Information about a postgres installation, which uniquely identifies the
    binary/git worktree that should be used. (but not which database files)
    """
    branch: PgBranch
    block_size: int = DEFAULT_BLOCK_SIZE
    bg_size: int = DEFAULT_BG_SIZE

    @property
    def builddir(self) -> str:
        if self.branch.is_pbm:
            return f'{self.branch.name}_blksz{self.block_size}_bgsz{self.bg_size}'
        else:
            return f'base_blksz{self.block_size}'

    @property
    def build_path(self) -> Path:
        return POSTGRES_BUILD_PATH / self.builddir

    @property
    def install_path(self) -> Path:
        return POSTGRES_INSTALL_PATH / self.builddir


@dataclass(frozen=True)
class DbData:
    """
    Information which uniquely identifies the database data directory to use.
    """
    workload: Workload
    sf: int
    block_size: int = DEFAULT_BLOCK_SIZE

    @property
    def data_path(self) -> Path:
        return PG_DATA_ROOT / f'pg_{self.workload.name.lower()}_sf{self.sf}_blksz{self.block_size}'

    @property
    def db_name(self) -> str:
        return self.workload.db_name(self.sf)

    def conn_str(self, host: str) -> str:
        return f'pq://{host}/{self.db_name}'


@dataclass(frozen=True)
class DbConfig:
    """
    Information about the database is configured: branch/code being used, block size,
    and scale factor. These are used to decide which binaries to use (branch, block_size, bg_size)
    and which database cluster to start (block_size, sf, workload).
    """
    bin: DbBin
    data: DbData

    @property
    def branch(self) -> PgBranch:
        return self.bin.branch

    @property
    def sf(self) -> int:
        return self.data.sf

    @property
    def block_size(self) -> int:
        return self.bin.block_size

    @property
    def bg_size(self) -> int:
        return self.bin.bg_size

    def to_config_key(self) -> ConfigKey:
        return ConfigKey(sf=self.data.sf, blk_sz=self.bin.block_size)

    def to_config_map(self) -> dict:
        return {
            'branch': self.bin.branch.name,
            'scalefactor': self.data.sf,
            'block_size': self.bin.block_size,
            'block_group_size': self.bin.bg_size,
        }

    def check_consistent(self) -> bool:
        return self.bin.block_size == self.data.block_size


@dataclass
class CGroupConfig:
    mem_gb: float
    name: str = PG_CGROUP

    @property
    def mem_bytes(self):
        return int(self.mem_gb * 2**30)

    def to_config_map(self) -> dict:
        return {'cgroup_gb': self.mem_gb, }


@dataclass
class BBaseConfig:
    """
    Information about how to configure benchbase for the test, and any database setup that has to be run before the
    test which isn't persisted between runs (e.g. whether to prewarm)
    """
    nworkers: int
    workload: WorkloadConfig
    seed: int = 12345
    prewarm: bool = True

    def to_config_map(self) -> dict:
        return {
            'parallelism': self.nworkers,
            'prewarm': self.prewarm,
            'seed': self.seed,
            **self.workload.to_config_map(),
        }


@dataclass(frozen=True)
class RuntimePgConfig:
    """
    PostgreSQL configuration for the test that isn't relevant for which binary to use (branch and block size) or which
    database cluster (block size and scale factor). These get mapped directly to postgresql.conf so the field names
    should match the config field.
    Some of these are new and only supported for some branches, and should be `None` for the other branches
    """
    shared_buffers: str
    synchronize_seqscans: str = 'on'
    # PBM-only fields:
    # pbm2 and later:
    pbm_evict_num_samples: Optional[int] = None
    pbm_bg_naest_max_age: Optional[float] = None

    def config_dict(self) -> Dict[str, str]:
        return {k: (str(v) if v is not None else None) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class DbSetup:
    """Remember how a database is setup: what indexes and clustering?"""
    indexes: Optional[str]
    clustering: Optional[str]

    def update_with_old(self, old: 'DbSetup') -> 'DbSetup':
        ret = DbSetup(indexes=self.indexes or old.indexes,
                      clustering=self.clustering or old.clustering)
        return ret


@dataclass
class ExperimentConfig:
    """All configuration for an experiment"""
    pgconf: RuntimePgConfig
    dbconf: DbConfig
    dbsetup: DbSetup
    bbconf: BBaseConfig
    db_host: str = None
    cgroup: Optional[CGroupConfig] = None

    _res_dir: Optional[Path] = field(init=False, default=None)

    def __post_init__(self):
        if self.db_host is None:
            self.db_host = self.bbconf.workload.workload.default_db_host

    @property
    def results_dir(self) -> Path:
        if self._res_dir is not None:
            return self._res_dir

        # Create directory for results
        while True:
            # Directory is workload + time
            w_str = self.bbconf.workload.workload.name.upper()
            ts = dt.now()
            ts_str = ts.strftime('%Y-%m-%d_%H-%M')
            res_dir = RESULTS_ROOT / f'{w_str}_{ts_str}'
            try:
                os.makedirs(res_dir)
                break
            except FileExistsError:
                # if the file already exists, wait and try again with new timestamp
                print(f'WARNING: trying to save results to {res_dir} but it already exists! retrying...')
                time.sleep(15)

        self._res_dir = res_dir
        return self._res_dir

    @property
    def results_bbase_subdir(self) -> Path:
        return self.results_dir / f'{self.dbconf.branch.name}_blksz{self.dbconf.block_size}'


def read_last_config() -> DefaultDict[ConfigKey, DbSetup]:
    """
    Reads the current database index/cluser configuration from `LAST_CONFIG_FILE` and returns it as a dictionary.
    Keys are `ConfigKey`
    Values are `DbSetup`
    """
    try:
        with open(LAST_CONFIG_FILE, 'r') as cf:
            decoded = json.JSONDecoder().decode(cf.read())
    except FileNotFoundError:
        decoded = []

    last_conf = defaultdict(lambda: DbSetup(None, None))

    def field_names(c) -> List[str]:
        return [f.name for f in fields(c)]

    for d in decoded:
        d_k = ConfigKey(**{k: v for k, v in d.items() if k in field_names(ConfigKey)})
        d_v = DbSetup(**{k: v for k, v in d.items() if k in field_names(DbSetup)})

        last_conf[d_k] = d_v

    return last_conf


def update_last_config(conf: DbConfig, setup: DbSetup):
    """Update the configuration file for the specified config and setup."""
    last_conf = read_last_config()

    c = ConfigKey(sf=conf.sf, blk_sz=conf.block_size)
    last_conf[c] = setup

    to_encode = [{**asdict(k), **asdict(v)} for k, v in last_conf.items()]

    with open(LAST_CONFIG_FILE, 'w') as f:
        encoder = json.JSONEncoder(indent=2)
        f.write(encoder.encode(to_encode))


def get_last_config(conf: Union[DbConfig, ConfigKey]) -> DbSetup:
    """Get database setup from the config file for the given configuration."""
    last_conf = read_last_config()
    if isinstance(conf, DbConfig):
        return last_conf[conf.to_config_key()]
    else:
        return last_conf[conf]


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
    try:
        with GitProgressBar("PostgreSQL") as pbar:
            pg_repo = git.Repo.clone_from(POSTGRES_GIT_URL, POSTGRES_SRC_PATH_BASE, progress=pbar,
                                          multi_options=[f'--branch {BRANCH_POSTGRES_BASE.git_branch}'])
    except GitCommandError as e:
        print(f'WARNING: got git error while cloning main repo: {e}')
        print(f'    Assuming the repo is already cloned. Update it instead')

        pg_repo = git.Repo(POSTGRES_SRC_PATH_BASE)
        with GitProgressBar("PostgreSQL") as pbar:
            pg_repo.remote().pull(progress=pbar)

    print('Creating worktrees for other PostgreSQL branches')
    pbm_repos = []
    for branch in POSTGRES_PBM_BRANCHES:
        abs_dir = POSTGRES_SRC_PATH / branch.name
        try:
            print(f'Creating worktree for: {branch.name} under {abs_dir}')
            pg_repo.git.worktree('add', abs_dir, branch.git_branch)
        except GitCommandError as e:
            print(f'WARNING: got git error while creating worktree for {branch}: {e}')
            print(f'    Assuming the worktree already exists and continuing...')
        pbm_repos.append(git.Repo(abs_dir))


def clone_benchbase_repo():
    with GitProgressBar("BenchBase ") as pbar:
        bbase_repo = git.Repo.clone_from(BENCHBASE_GIT_URL, BENCHBASE_SRC_PATH, progress=pbar)
    return bbase_repo


def get_repos():
    """Return the already-cloned repositories created by `clone_repos`."""
    pg_repo = git.Repo(POSTGRES_SRC_PATH_BASE)
    pbm_repos = []
    for brnch in POSTGRES_PBM_BRANCHES:
        abs_dir = POSTGRES_SRC_PATH / brnch.name
        pbm_repos.append(git.Repo(abs_dir))
    bbase_repo = git.Repo(BENCHBASE_SRC_PATH)

    return pg_repo, pbm_repos, bbase_repo


def pbm_blk_shift(blk_sz: int, bg_size: int):
    blks_per_group = bg_size // blk_sz
    res = 0
    while blks_per_group > 1:
        res += 1
        blks_per_group //= 2
    return res


def config_postgres_repo(dbbin: DbBin):
    """Runs `./configure` to setup the build for postgres with the provided branch and block size."""
    build_path = dbbin.build_path
    install_path = dbbin.install_path
    brnch = dbbin.branch
    blk_sz = dbbin.block_size
    bg_size = dbbin.bg_size

    print(f'Configuring postgres {brnch.name} with block size {blk_sz}, block group size {bg_size}')
    build_path.mkdir(exist_ok=True, parents=True)

    if brnch.is_pbm:
        version_str = f'--with-extra-version=-{brnch.name}_blkzs{blk_sz}_bgsz{bg_size}'
    else:
        version_str = f'--with-extra-version=-{brnch.name}_blkzs{blk_sz}'

    config_args = [
        POSTGRES_SRC_PATH / brnch.name / 'configure',
        f'--with-blocksize={blk_sz}',
        f'--prefix={install_path}',
        version_str,
    ]

    # for PBM branches, need some extra config args
    if brnch.is_pbm:
        bg_shift = pbm_blk_shift(blk_sz, bg_size)
        config_args.append(f'--with-pbmblockshift={bg_shift}')

    subprocess.Popen(config_args, cwd=build_path).wait()


def build_postgres_extension(build_path: Path, extension: str):
    """Compile the given extension for the specified build path"""
    ext_build_path = build_path / 'contrib' / extension
    print(f'Compiling and installing extension {extension}')
    ret = subprocess.Popen('make', cwd=ext_build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when compiling extension {extension}')
    ret = subprocess.Popen(['make', 'install'], cwd=ext_build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when installing extension {extension}')


def build_postgres(dbbin: DbBin):
    """Compiles PostgreSQL for the specified branch/block size."""
    build_path = dbbin.build_path
    brnch = dbbin.branch
    blk_sz = dbbin.block_size
    bg_sz = dbbin.bg_size
    print(f'Compiling & installing postgres {brnch.name} with block size {blk_sz} and block group size {bg_sz}')
    ret = subprocess.Popen('make', cwd=build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when compiling postgres {brnch.name} with block size={blk_sz}, group size={bg_sz}')
    ret = subprocess.Popen(['make', 'install'], cwd=build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when installing postgres {brnch.name} with block size={blk_sz}, group size={bg_sz}')

    # compile desired extensions...
    for ext in ['pg_prewarm']:
        build_postgres_extension(build_path, ext)


def clean_postgres(dbbin: DbBin):
    """Clean PostgreSQL build for the specified branch/block size."""
    build_path = dbbin.build_path
    brnch = dbbin.branch
    print(f'Cleaning postgres {brnch.name} with block size {dbbin.block_size} and block group size {dbbin.bg_size}')
    ret = subprocess.Popen(['make', 'clean'], cwd=build_path).wait()
    if ret != 0:
        raise Exception(f'Got return code {ret} when cleaning postgres {brnch.name} with block size={dbbin.block_size}, group size={dbbin.bg_size}')


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
    pgi = pg.installation.Installation(pg_config_dictionary(case.bin.install_path / 'bin' / 'pg_config'))
    cl = Cluster(pgi, case.data.data_path)
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
        'work_mem': PG_WORK_MEM,
    })


def config_remote_postgres(conn: fabric.Connection, dbconf: DbConfig, pgconf: RuntimePgConfig):
    """Configure a PostgreSQL installation from a different host (the benchmark client).
    This configures it with essentially no security at all; we assume the host is not accessible
    to the general internet. (in any case, there is nothing on the database except test data)
    """
    local_temp_path = 'temp_pg.conf'
    remote_path = str(dbconf.data.data_path / 'postgresql.conf')

    # print(f'Configuring PostgreSQL on remote host, config file at: {remote_path}')

    conn.get(remote_path, local_temp_path)
    cf = ConfigFile(local_temp_path)

    cf.update({
        'listen_addresses': '*',
        'port': PG_PORT,
        'work_mem': PG_WORK_MEM,
        **pgconf.config_dict(),
    })

    conn.put(local_temp_path, remote_path)
    os.remove(local_temp_path)


def start_remote_postgres(conn: fabric.Connection, case: DbConfig, cgroup: CGroupConfig = None):
    """Start PostgreSQL from the benchmark client machine."""
    install_path = case.bin.install_path
    pgctl = install_path / 'bin' / 'pg_ctl'
    data_dir = case.data.data_path
    logfile = data_dir / 'logfile'

    conn.run(f'truncate --size=0 {logfile}')
    start_cmd = f'{pgctl} start -D {data_dir} -l {logfile}'
    if cgroup is not None:
        conn.run(f'cgset -r memory.limit_in_bytes={cgroup.mem_bytes} {cgroup.name}')
        start_cmd = f'cgexec -g memory:{cgroup.name} {start_cmd}'
    conn.run(start_cmd)


def stop_remote_postgres(conn: fabric.Connection, case: DbConfig, immediate=False):
    """Stop PostgreSQL remotely."""
    install_path = case.bin.install_path
    pgctl = install_path / 'bin' / 'pg_ctl -t 600'
    data_dir = case.data.data_path

    extra_args = '-m i' if immediate else ''

    conn.run(f'{pgctl} stop -D {data_dir} {extra_args}')


def get_remote_disk_stats(conn: fabric.Connection, case: DbConfig):
    """Get disk stats (sectors_read, sectors_written) of the remote host"""
    dev = case.data.workload.device
    res = conn.run(f'cat /sys/block/{dev}/stat', hide=True).stdout.split()
    return int(res[2]), int(res[6])


def prewarm_lineitem(data: DbData, dbhost: str):
    """Prewarm lineitem cache for the given database config (DB must be running)"""
    with pg.open(data.conn_str(dbhost)) as conn:
        conn: PgConnection
        conn.execute('CREATE EXTENSION IF NOT EXISTS pg_prewarm;')
        conn.execute('''select pg_prewarm('lineitem');''')


def clear_pg_stats(data: DbData, dbhost: str):
    """Clear IO statistics for the given database config (DB must be running))"""
    with pg.open(data.conn_str(dbhost)) as conn:
        conn: PgConnection
        conn.execute('SELECT pg_stat_reset();')


def create_bbase_config(sf: int, bb_config: BBaseConfig, out, host):
    """Set connection information and scale factor in a BenchBase config file."""
    tree = ET.parse(bb_config.workload.workload.base_config_file)
    db_name = bb_config.workload.workload.db_name(sf)
    app_name = bb_config.workload.workload.name

    params = tree.getroot()
    params.find('url').text = f'jdbc:postgresql://{host}:{PG_PORT}/{db_name}?sslmode=disable&amp;ApplicationName={app_name}&amp;reWriteBatchedInserts=true'
    params.find('username').text = PG_USER
    params.find('password').text = PG_PASSWD
    params.find('scalefactor').text = str(sf)
    params.find('terminals').text = str(bb_config.nworkers)
    params.find('randomSeed').text = str(bb_config.seed)

    # If applicable, add selectivity (making sure there isn't already a value for it first)
    for sel in params.findall('selectivity'):
        params.remove(sel)

    selectivity = bb_config.workload.selectivity
    if selectivity is not None:
        ET.SubElement(params, 'selectivity').text = str(selectivity)

    # Specify the workload
    works = params.find('works')
    for elem in works:
        works.remove(elem)

    work = ET.SubElement(works, 'work')

    ET.SubElement(work, 'serial').text = 'false'
    bb_config.workload.write_bbase_config(work)

    tree.write(out)


def run_bbase_load(b: str, config: Path, create=False):
    """Run BenchBase to load data with the given config file path."""

    ops = ['--load=true']
    if create:
        ops.append('--create=true')

    subprocess.Popen([
        'java',
        '-jar', str(BENCHBASE_INSTALL_PATH / 'benchbase-postgres' / 'benchbase.jar'),
        '-b', b,
        '-c', str(config),
        *ops,
    ], cwd=BENCHBASE_INSTALL_PATH / 'benchbase-postgres').wait()


def run_bbase_test(exp: ExperimentConfig):
    """
    Run benchbase (on local machine) against PostgreSQL on the remote host.
    Will start & stop PostgreSQL on the remote host.
    Returns (sectors read, sectores written)
    """
    dbconf = exp.dbconf
    bbconf = exp.bbconf
    pgconf = exp.pgconf

    workload = bbconf.workload.workload
    db_host = exp.db_host
    bb_workload_name = workload.name.lower()
    temp_bbase_config = BUILD_ROOT / f'bbase_{bb_workload_name}_config.xml'

    # sanity checks
    assert dbconf.check_consistent(), "Error with the DB configuration!"
    assert workload == dbconf.data.workload, "BB config and DB confid have different workloads!"

    create_bbase_config(dbconf.sf, bbconf, temp_bbase_config, host=db_host)

    with FabConnection(db_host) as conn:
        config_remote_postgres(conn, dbconf, pgconf)
        start_remote_postgres(conn, dbconf, cgroup=exp.cgroup)
        # empty the buffer cache on remote host
        conn.run('echo 1 | sudo tee /proc/sys/vm/drop_caches')

    try:
        # prewarm lineitem table if desired (TPCH only)
        if bbconf.prewarm and workload is TPCH:
            print(f'Pre-warming cache for lineitem table...')
            prewarm_lineitem(dbconf.data, db_host)

        # Clear statistics on remote postgres
        clear_pg_stats(dbconf.data, db_host)

        # get iostats! (/sys/blocks/sdb/stat in this case)
        with FabConnection(db_host) as conn:
            pre_reads, pre_writes = get_remote_disk_stats(conn, dbconf)

        # Run benchbase
        subprocess.Popen([
            'java',
            '-jar', str(BENCHBASE_INSTALL_PATH / 'benchbase-postgres' / 'benchbase.jar'),
            '-b', bb_workload_name,
            '-c', str(temp_bbase_config),
            '--execute=true',
            '-d', str(exp.results_bbase_subdir),
        ], cwd=BENCHBASE_INSTALL_PATH / 'benchbase-postgres').wait()

        # get & return iostats after the test
        with FabConnection(db_host) as conn:
            post_reads, post_writes = get_remote_disk_stats(conn, dbconf)

        return (post_reads - pre_reads), (post_writes - pre_writes)

    finally:
        with FabConnection(db_host) as conn:
            stop_remote_postgres(conn, dbconf, immediate=workload is TPCC)


def pg_exec_file(conn: PgConnection, file):
    with open(file, 'r') as f:
        stmts = ''.join(f.readlines())
    conn.execute(stmts)


def create_and_populate_tpch_local(case: DbConfig):
    """
    Initialize a database with TPCH data for the given test case.
    Note: we only need to initialize for the base branch since each branch can use the same data dir
    """
    db_name = case.data.db_name
    create_ddl_file = 'ddl/create-tables-noindex.sql'
    conn_str = f'pq://localhost/{db_name}'

    cl = pg_get_cluster(case)

    print(f'(Re-)Initializing database cluster at {cl.data_directory}...')
    shutil.rmtree(cl.data_directory, ignore_errors=True)
    pg_init_local_db(cl)

    print(f'Starting cluster and creating database {db_name} with tables (defined in {create_ddl_file}) on local host...')
    pg_start_db(cl)
    try:
        subprocess.run([case.bin.install_path / 'bin' / 'createdb', db_name])
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
        bbconf = BBaseConfig(nworkers=1, workload=WORKLOAD_TPCH_WEIGHTS)
        create_bbase_config(case.sf, bbconf, bbase_config_file, host='localhost')
        run_bbase_load('tpch', bbase_config_file)

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


def tpcc_src_data_dir(case: DbConfig) -> Path:
    """Name of directory to copy TPCC data so we can copy it instead of re-generating every time."""
    return PG_DATA_ROOT / f'src_tpcc_sf{case.sf}_blksz{case.block_size}'


def create_and_populate_tpcc_local(case: DbConfig):
    """
    Initialize a database with TPCC data for the given test case.
    Note: we only need to initialize for the base branch since each branch can use the same data dir
    """
    db_name = f'TPCC_{case.sf}'
    conn_str = f'pq://localhost/{db_name}'

    cl = pg_get_cluster(case)

    print(f'(Re-)Initializing database cluster at {cl.data_directory}...')
    shutil.rmtree(cl.data_directory, ignore_errors=True)
    pg_init_local_db(cl)

    print(f'Starting cluster for benchbase load {db_name}')
    pg_start_db(cl)
    try:
        subprocess.run([case.bin.install_path / 'bin' / 'createdb', db_name])

        print(f'BenchBase: loading test data...')
        bbase_config_file = BUILD_ROOT / 'load_config.xml'
        bbconf = BBaseConfig(nworkers=1, workload=WORKLOAD_TPCC)
        create_bbase_config(case.sf, bbconf, bbase_config_file, host='localhost')
        run_bbase_load('tpcc', bbase_config_file, create=True)

        # Run ANALYZE to make sure stats are up-to-date, and CHECKPOINT to clear out the WAL
        with pg.open(conn_str) as conn:
            conn.execute('ANALYZE;')
            conn.execute('CHECKPOINT;')

    finally:
        print(f'Shutting down database cluster {cl.data_directory}...')
        pg_stop_db(cl)

    # Rename the database file. TPCC modifies the database, so we need a way to
    # reset at the start of each test.
    alt_dir = tpcc_src_data_dir(case)
    print(f'Moving {cl.data_directory} to {alt_dir}...')

    shutil.rmtree(alt_dir, ignore_errors=True)
    shutil.move(cl.data_directory, alt_dir)


def tpcc_restore_data_dir(case: DbConfig):
    """Copy the TPCC data directory back into place on the remote host."""
    data_dir = case.data.data_path
    src_dir = tpcc_src_data_dir(case)

    print(f'Restoring TPCC database files: copying {src_dir} to {data_dir}...')
    with FabConnection(case.data.workload.default_db_host) as fabconn:
        fabconn.run(f'rm --recursive --force {data_dir}')  # --force to ignore error if it is already not there
        fabconn.run(f'cp --recursive {src_dir} {data_dir}')


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
    """return the constraints and indexes as dataframes"""
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


def setup_indexes_cluster_tpch(blk_sz: int, sf: int, db_host: str, *, prev: DbSetup, new: DbSetup):
    """Change indexes and clustering on the database. Remembers the changes in `last_config.json`"""

    with FabConnection(db_host) as fabconn:
        dbbin = DbBin(branch=BRANCH_POSTGRES_BASE, block_size=blk_sz)
        dbdata = DbData(workload=TPCH, sf=sf, block_size=blk_sz)
        dbconf = DbConfig(bin=dbbin, data=dbdata)
        # Use large amount of memory for creating indexes
        config_remote_postgres(fabconn, dbconf, RuntimePgConfig(shared_buffers='20GB', synchronize_seqscans='on'))
        start_remote_postgres(fabconn, dbconf)

        try:
            with pg.open(f'pq://{db_host}/TPCH_{sf}') as pgconn:
                reconfigure_indexes(pgconn, blk_sz, prev_indexes=prev.indexes, new_indexes=new.indexes)
                reconfigure_clustering(pgconn, prev_cluster=prev.clustering, new_cluster=new.clustering or prev.clustering)

                ret = read_constraints_indexes(pgconn)

            # remember the changes
            update_last_config(dbconf, new)

        finally:
            stop_remote_postgres(fabconn, dbconf)

    return ret


def one_time_pg_setup():
    """Gets postgres installed for each version that is needed.
    Must be run once on the postgres server host
    """

    print('Cloning PostreSQL repo & worktrees...')
    clone_pg_repos()

    # Compile postgres for each different version

    for blk_sz in PG_BLK_SIZES:
        dbbin = DbBin(branch=BRANCH_POSTGRES_BASE, block_size=blk_sz)
        config_postgres_repo(dbbin)
        build_postgres(dbbin)

    for brnch in POSTGRES_PBM_BRANCHES:
        for blk_sz in PG_BLK_SIZES:
            for bg_size in BLOCK_GROUP_SIZES:
                dbbin = DbBin(branch=brnch, block_size=blk_sz, bg_size=bg_size)
                config_postgres_repo(dbbin)
                build_postgres(dbbin)


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
        for brnch in POSTGRES_ALL_BRANCHES:
            # touch PBM related files to reduce chance of needing to clean and fully rebuild...
            if brnch.is_pbm:
                incl_path = POSTGRES_SRC_PATH / brnch.name / 'src' / 'include' / 'storage'
                (incl_path / 'pbm.h').touch(exist_ok=True)

                src_path = POSTGRES_SRC_PATH / brnch.name / 'src' / 'backend' / 'storage' / 'buffer'
                (src_path / 'pbm.c').touch(exist_ok=True)
                (src_path / 'pbm_internal.c').touch(exist_ok=True)
                (src_path / 'freelist.c').touch(exist_ok=True)
                (src_path / 'bufmgr.c').touch(exist_ok=True)

                for bg_size in BLOCK_GROUP_SIZES:
                    dbbin = DbBin(brnch, block_size=blk_sz, bg_size=bg_size)
                    build_postgres(dbbin)

            else:
                dbbin = DbBin(brnch, block_size=blk_sz)
                build_postgres(dbbin)


def clean_pg_installs(base=False):
    """Clean postgres installations. (only include the base branch if base=True)
    Run on the server host.
    """

    for brnch in POSTGRES_ALL_BRANCHES:
        for blk_sz in PG_BLK_SIZES:
            if brnch.is_pbm:
                for bg_sz in BLOCK_GROUP_SIZES:
                    dbbin = DbBin(brnch, block_size=blk_sz, bg_size=bg_sz)
                    clean_postgres(dbbin)
            elif base:
                dbbin = DbBin(brnch, block_size=blk_sz)
                clean_postgres(dbbin)


def one_time_benchbase_setup():
    """Build and install BenchBase on current host."""
    print('Cloning BenchBase...')
    clone_benchbase_repo()
    build_benchbase()
    install_benchbase()


def gen_data_tpch(sf: int, blk_sz: int):
    if sf is None:
        raise Exception(f'Must specify scale factor when loading data!')

    # Generate test data (only base branch is needed for generating data)
    print('--------------------------------------------------------------------------------')
    print(f'---- Initializing TPCH data for blk_sz={blk_sz}, sf={sf}')
    print('--------------------------------------------------------------------------------')
    dbbin = DbBin(BRANCH_POSTGRES_BASE, block_size=blk_sz)
    dbdata = DbData(TPCH, sf=sf, block_size=blk_sz)
    dbconf = DbConfig(dbbin, dbdata)
    create_and_populate_tpch_local(dbconf)


def gen_data_tpcc(sf: int, blk_sz: int):
    if sf is None:
        raise Exception(f'Must specify scale factor (# of warehouses) when loading data!')

    # Generate test data (only base branch is needed for generating data)
    print('--------------------------------------------------------------------------------')
    print(f'---- Initializing TPCC data for blk_sz={blk_sz}, sf={sf}')
    print('--------------------------------------------------------------------------------')
    dbbin = DbBin(BRANCH_POSTGRES_BASE, block_size=blk_sz)
    dbdata = DbData(TPCC, sf=sf, block_size=blk_sz)
    dbconf = DbConfig(dbbin, dbdata)
    create_and_populate_tpcc_local(dbconf)


def drop_all_indexes_tpch(sf: int, blk_sz: int, db_host: str):
    """Drop all indexes and constraints. More powerful cleanup function if something goes really wrong."""
    if sf is None:
        raise Exception(f'Must specify scale factor of databases to clean up!')

    db_host = db_host or TPCH.default_db_host

    print(f'~~~~~~~~~~ Dropping all indexes and constraints for TPCH blk_sz={blk_sz}, sf={sf} ~~~~~~~~~~')

    dbbin = DbBin(BRANCH_POSTGRES_BASE, block_size=blk_sz)
    dbdata = DbData(TPCH, sf=sf, block_size=blk_sz)
    dbconf = DbConfig(dbbin, dbdata)
    with FabConnection(db_host) as fabconn:
        config_remote_postgres(fabconn, dbconf, RuntimePgConfig(shared_buffers='20GB', synchronize_seqscans='on'))
        start_remote_postgres(fabconn, dbconf)

        try:
            with pg.open(dbdata.conn_str(db_host)) as pgconn:
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

                update_last_config(dbconf, DbSetup(indexes=None, clustering=None))

        finally:
            stop_remote_postgres(fabconn, dbconf)

    print(f'~~~~~~~~~~ All indexes have been dropped ~~~~~~~~~~')


def reindex(args):
    """Change the clustering of the database without running benchmarks."""
    if args.sf is None:
        raise Exception(f'Must specify scale factor of databases to recluster!')

    # read in what indexes/clustering is currently used in the database
    # last_config = read_last_config()
    prev_setup = get_last_config(ConfigKey(sf=args.sf, blk_sz=args.blk_sz))

    # special case: if 'none' is specified we want to forget what the clustering is (and do nothing)
    # in this case set new_cluster to None (handled by setup_indexes_cluster, which updates our state file)
    # if args.cluster is none, it means it was not specified so stay with previous clustering
    forget_clustering = (args.cluster is not None and args.cluster.lower() == 'none')
    if forget_clustering:
        new_cluster = None
    else:
        new_cluster = args.cluster or prev_setup.clustering
    new_indexes = args.index_type or prev_setup.indexes
    new_setup = DbSetup(indexes=new_indexes, clustering=new_cluster)

    print(f'~~~~~~~~~~ Reconfiguring using indexes {new_indexes}, clustering {new_cluster} for blk_sz={args.blk_sz}, sf={args.sf} ~~~~~~~~~~')
    setup_indexes_cluster_tpch(args.blk_sz, args.sf, db_host=args.host or PG_HOST_TPCH, prev=prev_setup, new=new_setup)


def run_experiment(experiment: str, exp_config: ExperimentConfig):
    dbconf = exp_config.dbconf
    bbconf = exp_config.bbconf
    pgconf = exp_config.pgconf
    dbsetup = exp_config.dbsetup

    sf = dbconf.sf
    blk_sz = dbconf.block_size
    is_tpch = (bbconf.workload.workload is TPCH)

    # Check current status of indexes & clustering in the database
    if is_tpch:
        prev_setup: DbSetup = get_last_config(dbconf)
        dbsetup = dbsetup.update_with_old(prev_setup)
        dbsetup_dict = asdict(dbsetup)
    else:
        dbsetup_dict = {}

    # Write configuration to a file
    with open(exp_config.results_dir / CONFIG_FILE_NAME, 'w') as f:
        config = {
            'experiment': experiment,
            'work_mem': PG_WORK_MEM,
            **dbconf.to_config_map(),
            **asdict(pgconf),
            **bbconf.to_config_map(),
            **exp_config.cgroup.to_config_map(),
            **dbsetup_dict,
        }
        if exp_config.cgroup is not None:
            config.update(exp_config.cgroup.to_config_map())
        f.write(json.JSONEncoder(indent=2, sort_keys=True).encode(config))
        f.write('\n')  # ensure trailing newline

    results_dir = exp_config.results_dir

    # Print out summary to the console
    print(f'======================================================================')
    print(f'== Running experiments with:')
    print(f'==   Seed:                  {bbconf.seed}')
    print(f'==   DB host:               {exp_config.db_host}')
    print(f'==   Branch:                {dbconf.branch.name}')
    # print(f'==   Scale factor:          {sf}')
    # print(f'==   Block size:            {blk_sz} KiB')
    # print(f'==   Block group size       {dbconf.bg_size} KiB')
    # print(f'==   Workload:              {bbconf.workload.workname}')
    # print(f'==   Worker memory          {PG_WORK_MEM}')
    print(f'==   Shared memory:         {pgconf.shared_buffers}')
    if is_tpch:
        print(f'==   Index definitions:     ddl/index/{dbsetup.indexes}/')
        print(f'==   Clustering:            dd/cluster/{dbsetup.clustering}.sql')
    if isinstance(bbconf.workload, WeightedWorkloadConfig):
        t = bbconf.workload.time_s
        s = t % 60
        s = '' if s == 0 else f' {s} s'
        print(f'==   Time:                  {t // 60} min{s}')
    print(f'==   Terminals:             {bbconf.nworkers}')
    print(f'==   PBM num samples:       {pgconf.pbm_evict_num_samples}')
    print(f'== Storing results to {results_dir}')
    print(f'======================================================================')

    if is_tpch:
        # Make sure we have the desired indexes & clustering if applicable
        print(f'~~~~~~~~~~ Setup indexes={dbsetup.indexes}, clustering={dbsetup.clustering} for blk_sz={blk_sz}, sf={sf} ~~~~~~~~~~')
        constraints, indexes = setup_indexes_cluster_tpch(blk_sz, sf, exp_config.db_host,
                                                          prev=prev_setup, new=dbsetup)

        # remember when indexes and constraints are defined in case we want to double check later...
        with open(results_dir / CONSTRAINTS_FILE, 'w') as f:
            constraints.to_csv(f, index=False)

        with open(results_dir / INDEXES_FILE, 'w') as f:
            indexes.to_csv(f, index=False)

        print(f'~~~~~~~~~~ Index and clustering setup done! Running the real tests... ~~~~~~~~~~')
    else:
        # for TPCC: need to copy the database file!
        tpcc_restore_data_dir(dbconf)

    # Actually run the tests
    reads, writes = run_bbase_test(exp_config)
    rename_bbase_results(exp_config.results_bbase_subdir)
    # store IO stats in the results
    with open(exp_config.results_bbase_subdir / IOSTATS_FILE, 'w') as f:
        iostats = {'sectors_read': reads, 'sectors_written': writes, }
        f.write(json.JSONEncoder(indent=2, sort_keys=True).encode(iostats))
        f.write('\n')  # ensure trailing newline

    print(f'disk reads = {reads} = {reads * 512 / 2**20} MiB = {reads * 512 / 2**30} GiB')


def run_bench(args):
    experiment: str = args.experiment
    branch: PgBranch
    sf: int = args.sf

    if args.branch is None or args.branch not in (b.name for b in POSTGRES_ALL_BRANCHES):
        raise Exception(f'Must specify branch! Unrecognized value {args.branch}')
    branch = {b.name: b for b in POSTGRES_ALL_BRANCHES}[args.branch]

    if sf is None:
        raise Exception(f'Must specify scale factor!')

    if args.workload not in WORKLOADS_MAP:
        raise Exception(f'Unknown workload type {args.workload}')
    workload = WORKLOADS_MAP[args.workload]

    db_host = args.host or workload.workload.default_db_host

    if args.selectivity is not None:
        workload = workload.with_selectivity(args.selectivity)

    if isinstance(workload, CountedWorkloadConfig):
        workload = workload.with_multiplier(args.count_multiplier)

    num_samples = args.num_samples
    if branch.accepts_nsamples and num_samples is None:
        # default number of samples for branches which support it
        num_samples = 10

    pgconf = RuntimePgConfig(
        shared_buffers=args.shmem,
        synchronize_seqscans='on' if args.syncscans else 'off',
        pbm_evict_num_samples=num_samples,
    )

    dbbin = DbBin(branch, block_size=args.blk_sz, bg_size=args.bg_sz)
    dbdata = DbData(workload.workload, sf=sf, block_size=args.blk_sz)
    dbconf = DbConfig(dbbin, dbdata)
    dbsetup = DbSetup(indexes=args.index_type,
                      clustering=args.cluster)
    bbconf = BBaseConfig(nworkers=args.parallelism, workload=workload, prewarm=args.prewarm)

    # Actually run the experiment after parsing args
    exp = ExperimentConfig(pgconf=pgconf, dbconf=dbconf, dbsetup=dbsetup, bbconf=bbconf, db_host=db_host)
    run_experiment(experiment, exp)

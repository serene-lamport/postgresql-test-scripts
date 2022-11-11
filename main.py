#!/usr/bin/env python3
import git
import shutil
import subprocess
from pathlib import Path

import tqdm


###################
#  CONFIGURATION  #
###################

# path to clone stuff into (absolute)
ROOT_DIR = Path('/home/ta3vande/test_scripts/TESTING')

POSTGRES_GIT_URL = 'ist-git@git.uwaterloo.ca:ta3vande/postgresql-masters-work.git'
POSTGRES_SRC_PATH = ROOT_DIR / 'pg_src'
POSTGRES_SRC_PATH_BASE = POSTGRES_SRC_PATH / 'base'
POSTGRES_BASE_BRANCH = 'REL_14_STABLE'
POSTGRES_PBM_BRANCHES = {
    'pbm1': 'pbm_part1'
}
POSTGRES_ALL_BRANCHES = POSTGRES_PBM_BRANCHES.copy()
POSTGRES_ALL_BRANCHES['base'] = POSTGRES_BASE_BRANCH

POSTGRES_BUILD_PATH = ROOT_DIR / 'pg_build'
POSTGRES_INSTALL_PATH = ROOT_DIR / 'pg_install'

PG_BLK_SIZES = [8, 32]
# PG_BLK_SIZES = [8]


BENCHBASE_GIT_URL = 'https://github.com/cmu-db/benchbase.git'
BENCHBASE_SRC_PATH = ROOT_DIR / 'benchbase_src'
BENCHBASE_INSTALL_PATH = ROOT_DIR / 'benchbase_install'


class GitProgressBar(git.RemoteProgress):
    """Progress bar for git operations"""
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
    pg_repo = git.Repo(POSTGRES_SRC_PATH_BASE)
    pbm_repos = []
    for folder in POSTGRES_PBM_BRANCHES.keys():
        abs_dir = POSTGRES_SRC_PATH / folder
        pbm_repos.append(git.Repo(abs_dir))
    bbase_repo = git.Repo(BENCHBASE_SRC_PATH)

    return pg_repo, pbm_repos, bbase_repo


def get_build_path(folder, blk_sz):
    return POSTGRES_BUILD_PATH / f'{folder}_{blk_sz}'


def get_install_path(folder, blk_sz):
    return POSTGRES_INSTALL_PATH / f'{folder}_{blk_sz}'


def config_postgres():
    for folder in POSTGRES_ALL_BRANCHES.keys():
        for blk_sz in PG_BLK_SIZES:
            build_path = get_build_path(folder, blk_sz)
            install_path = get_install_path(folder, blk_sz)
            print(f'Configuring postgres {folder} with block size {blk_sz}')
            build_path.mkdir(exist_ok=True, parents=True)
            subprocess.Popen([
                POSTGRES_SRC_PATH / folder / 'configure',
                f'--with-blocksize={blk_sz}',
                f'--prefix={install_path}'
            ], cwd=build_path).wait()


def build_postgres():
    for folder in POSTGRES_ALL_BRANCHES.keys():
        for blk_sz in PG_BLK_SIZES:
            build_path = get_build_path(folder, blk_sz)
            print(f'Compiling & installing postgres {folder} with block size {blk_sz}')
            ret = subprocess.Popen('make', cwd=build_path).wait()
            if ret != 0:
                raise Exception(f'Got return code {ret} when compiling postgres {folder} with block size={blk_sz}')
            subprocess.Popen(['make', 'install'], cwd=build_path).wait()


def build_benchbase():
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


def run_benchbase(config_file):
    pass

    # TODO!



if __name__ == '__main__':
    # (pg_repo, pbm_repos, bbase_repo) = clone_repos()

    # config_postgres()
    # build_postgres()

    # build_benchbase()
    install_benchbase()

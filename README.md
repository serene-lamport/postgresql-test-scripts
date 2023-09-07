PostgreSQL test scripts
=======================

Note: `first_time_setup.sh` automates some of this (installing apt packages, creating the cgroups)


Prerequisites
-------------

- Python 3 >= 3.7
- java jdk 17 (or newer) to compile & run benchbase: `sudo apt install openjdk-17-jdk`
- prerequisites to build PostgreSQL
- `python3-tk` for the `results_ploy.py` script to show figures
- `cgroup-tools` for creating and configuring cgroups to limit total system memory, including availble OS cache space


Python virtual environment
--------------------------

1. Install virtualenv with `pip install virtualenv` (or `python3 -m pip` if `pip` is not found)
2. Create the virtual environment: `virtualenv -p python3 venv` (inside this repo)
3. Activate the environment: `source script_env/bin/activate` (deactivate later by running `deactivate`)
4. Install python libraries once activated: `pip install -r requirements.txt`
5. After adding a new package with `pip install` (with the environment activated), use `pip freeze > requirements.txt` to update the list


Using the scripts
-----------------

1. Clone this repository and setup the virtual environment as above. Do this on all test machines. (postgres machine and workload generator) (on the test cluster I used: only need to do it once as the home directory is shared)
2. Edit `lib/config.py` if neccessary.
    - `*_HOST_ARGS*`: The host and directory where postgres is installed for the experiments.
        - `data_root`: Tuple of postgres data directory (sub-folders will be created for each database), and the device/partition on which the directory is located. Partition be determined with `df <dir>`. Path should be absolute, not relative.
        - `db_host`: Hostname/IP of the machine.
    - Also set `PG_USER` if the usernames aren't the same on each machine.
    - `BUILD_ROOT`: The directory where code will be compiled and binaries installed. This should be an absolute path.
3. Make sure `BUILD_ROOT` and the various `data_root` paths exist on the relevant hosts.
4. Run `sudo cgcreate -t $USER: -a $USER: -g memory:postgres_pbm` on the postgres host to create the cgroup used for testing. (a group can optionally be specified after `$USER:`) (this is already done by `first_time_setup.sh`)
5. Run `./run_util.py pg_setup` on the postgres machine to clone, build, and install postgress on all configurations.
6. Run `./run_util.py benchbase_setup` on both machines (or only one, if `BUILD_ROOT` is a shared network drive) to install benchbase. For me BenchBase can't compile on the test machines, so if this fails you'll have to compile it manually (on a different machine) and copy the files over. See the section below for more details.
7. Run `./run_util.py gen_test_data -sf <scalefactor>` on the postgres machine to load data through benchbase.
8. See `./run_util.py --help` for other tasks. May need to check the code for exactly what each does.
9. Running experiments: configure the experiments in `./run_experiments.py` and run it from the _test_ machine (not the postgres host).
    - See the comments block at the bottom of `run_experiments.py` for more details.
    - The script is called with one (or more) argument specifying the experiment(s) to run.
    - The experiments may run for a long time, so I highly recommend running the script in a `tmux` session (or something equivalent) so you don't need to stay connected to the machine.
10. After experiments complete, run `results_collect_to_csv.py` to aggregate the results in to `results.csv`.
    - Note: this will _replace_ `results.csv`, not retaining data for experiments where the raw results are not present on your machine. You may want to copy it first and modify the scripts in some way to retain the old data.
11. Modify and run `./results_plot.py` to generate graphs or do manual analysis.
    - See comments at the bottom for more about how this file works.
    - This scripts will give you a python shell when it is done: `Ctrl+D` or `exit()` to exit.
    - Specify `show` and/or `save` as command line args to show the plots and/or save them (as latex) to `BUILD_ROOT/figures`. (this can also be done manually from the shell)


Changing Benchbase
------------------
Due to the home directory on the test cluster being NFS (and possibly slightly misconfigured?), file locks don't work which prevent maven from working. Thus I compile BenchBase locally and copy the files separately, using `rebuild.sh` in the BenchBase repository. (Script may need to be adapted for your use)


Changing Postgres
-----------------
If the changes are in a new branch, it will have to be added to `config.py`, experiments will need to be updated to use it, `./run_util.py pg_setup` is needed to build the new branches. Otherwise, `./run_util.py pg_update` (after pushing changes to git) will pull down new changes and recompile. If the incremental build breaks something (when `make` doesn't realise some file needs to be recompiled), use `pg_clean` to force a full recompile.

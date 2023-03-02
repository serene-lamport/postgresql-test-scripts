PostgreSQL test scripts
=======================

Prerequisites
-------------

- Python 3 >= 3.7
- java jdk 17 (or newer) to compile & run benchbase: `sudo apt install openjdk-17-jdk`
- prerequisites to build PostgreSQL
- `python3-tk` for the process_results script to show figures
- `cgroup-tools` for creating and configuring cgroups to limit total system memory, including availble OS cache space


Python virtual environment
--------------------------

1. Install virtualenv with `pip install virtualenv` (or `python3 -m pip` if `pip` is not found)
2. Create the virtual environment: `virtualenv -p python3 venv`
3. Activate the environment: `source script_env/bin/activate` (deactivate later by running `deactivate`)
4. Install requirements once activated: `pip install -r requirements.txt`
5. After adding a new package with `pip install` (with the environment activated), use `pip freeze > requirements.txt` to update the list

Using the script
----------------

1. Clone this repository and setup the virtual environment as above. Do this on all test machines. (postgres machine and workload generator)
2. `./main.py --help` descripes the general operations. Before running, open `main.py` and edit the configuration variables at the top of the file. (make sure you have the same changes on each machine if you are not using network storage) The main variables to change are:
    - `PG_HOST` (and optionally `PG_PORT`: Hostname or IP of the machine where postgres should be installed/run during the experiments.
    - `PG_USER` (and optionally `PG_PASSWORD`, but by default it will be configured with no password): The username to connect as, should likely be your OS user used for the first-time setup.
    - `BUILD_ROOT`: The directory where code will be compiled and binaries installed. This should also be an absolute path.
    - `PG_DATA_ROOT`: The directory where postgres database files will be generated. Make sure there is lots of space here, and that this is an absolute path not relative.
    - `PG_DATA_DEVICE`: The device on the host where `PG_DATA_ROOT` is located. This can be determined with `df <PG_DATA_ROOT>`. (note: df will return the specific partition. Check the output of `lsblk` for the device name of the partition)

3. Create the directory pointed to by `PG_DATA_ROOT` and `BUILD_ROOT` on the relevant hosts.
4. Run `sudo cgcreate -t $USER: -a $USER: -g memory:postgres_pbm` on the postgres host to create the cgroup used for testing. (a group can optionally be specified after `$USER:`)
5. Run `./run_util.py pg_setup` on the postgres machine to clone, build, and install postgress on all configurations.
6. Run `./run_util.py benchbase_setup` on both machines (or only one, if `BUILD_ROOT` is a shared network drive) to install benchbase.
7. Run `./run_util.py gen_test_data -sf <scalefactor>` on the postgres machine to load data through benchbase.
8. Finally, run `./run_util.py bench ...` from the _test_ machine (not the postgres host) to run the benchmarks. See `./main.py --help` for arguments to `bench`.
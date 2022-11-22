PostgreSQL test scripts
=======================

Prerequisites
-------------

- Python 3 >= 3.7
- java jdk 17 (or newer) to compile & run benchbase: `sudo apt install opensdk-17-jdk`
- prerequisites to build PostgreSQL

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
    - `PG_DATA_ROOT`: The directory where postgres database files will be generated. Make sure there is lots of space here, and that this is an absolute path not relative.
    - `BUILD_ROOT`: The directory where code will be compiled and binaries installed. This should also be an absolute path.

3. Run `./main.py pg_setup` on the postgres machine to clone, build, and install postgress on all configurations.
4. Run `./main.py benchbase_setup` on both machines (or only one, if `BUILD_ROOT` is a shared network drive) to install benchbase.
5. Run `./main.py gen_test_data -sf <scalefactor>` on the postgres machine to load data through benchbase.
6. Finally, run `./main.py bench ...` from the _test_ machine (not the postgres host) to run the benchmarks. See `./main.py --help` for arguments to `bench`.
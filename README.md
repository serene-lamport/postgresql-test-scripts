PostgreSQL test scripts
=======================

Python virtual environment
--------------------------

1. Install virtualenv with `pip install virtualenv` (or `python3 -m pip` if `pip` is not found)
2. Create the virtual environment: `virtualenv -p python3 venv`
3. Activate the environment: `source script_env/bin/activate` (deactivate later by running `deactivate`)
4. Install requirements once activated: `pip install -r requirements.txt`
5. After adding a new package with `pip install` (with the environment activated), use `pip freeze > requirements.txt` to update the list



TODO
----
- [ ] ...
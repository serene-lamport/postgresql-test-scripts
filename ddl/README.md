# Indexes

To define indexes to use, create a subdirectory (under `ddl/index/`, e.g. `ddl/index/myindexes/` in this repository) with the following SQL scripts:
- `create.sql` should create all desired indexes and constraints to be used for the tests.
- `drop.sql` drops all indexes and constraints after the test run. Drop statements should always be `DROP INDEX IF EXISTS ...` since the drop script is always executed before create.
Then when calling the script pass the name of the directory to the index flag, e.g. `--index-type myindexes`


# Clustering

To specify how tables should be clustered create a sql script under `ddl/cluster/` which clusters the tables. This is run after the script to create indexes, so it can cluster by any of those indexes (assuming the index set specified has those indexes!). If clustering by something else, the script should both create and drop the index after clustering. (e.g. `CREATE INDEX myindex ON T ...; CLUSTER T USING myindex; DROP INDEX myindex;`) After clustering, the script should also call `ANALYZE <table>;` to re-analyze the re-organized table. When calling the script pass the name of the cluster script without the `.sql` extension, e.g. if the sql file is `ddl/cluster/myclustering.sql` pass `--cluster myclustering`.

Note that tables are not "unclustered" at the end - the same clustering can be used for multiple test runs and only needs to be specified once.
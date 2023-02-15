TODO
====
- [ ] Modify scripts to work for TPCC as well, possibly others?
- [ ] Alternate mode to pass in a % of table to benchbase for less variance with different #s of workers


TPCC support
------------
- [x] different host!
- [x] support creating database/test data - don't need to handle indexes (default ones are ideal)
- [x] therefore we don't need to update last_config, can stay as-is, but various things need to care about TPCH vs TPCC
- [x] different configuration for new workload...
- [x] running a test should restore a copy of the DB file... TPCC does updates with no undo option...
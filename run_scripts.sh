
# # brin to btree
# mv /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8 /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8_brin
# mv /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8_btree /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8
# cp /home/mkhalaji/pbm/postgresql-test-scripts/last_config_nasus_btree.json /home/mkhalaji/pbm/postgresql-test-scripts/last_config_nasus.json

# # btree to brin
# mv /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8 /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8_btree
# mv /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8_brin /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8
# cp /home/mkhalaji/pbm/postgresql-test-scripts/last_config_nasus_brin.json /home/mkhalaji/pbm/postgresql-test-scripts/last_config_nasus.json


# XXXX

# # run the sample sizes thing
# python3 run_experiments.py micro_seqscans

# # brin to btree
# mv /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8 /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8_brin
# mv /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8_btree /var/mkhalaji/pgdata/pg_tpch_sf50_blksz8
# cp /home/mkhalaji/pbm/postgresql-test-scripts/last_config_nasus_btree.json /home/mkhalaji/pbm/postgresql-test-scripts/last_config_nasus.json

# # run the trailing index thing
# python3 run_experiments.py micro_trailing_idx


# python3 run_experiments.py tpch


# wait until there's no process running with the name "run_experiments.py"
while pgrep -f "run_experiments.py" > /dev/null; do
    echo "waiting for run_experiments.py to finish"
    sleep 600
done

echo "run_experiments.py finished"
echo "Sleeping for 5 minutes"
sleep 300

while pgrep -f "run_experiments.py" > /dev/null; do
    echo "waiting for run_experiments.py to finish"
    sleep 300
done



mv bbase_config/sample_ycsb_config.xml bbase_config/sample_ycsb_config_0.99.xml
mv bbase_config/sample_ycsb_config_0.75.xml bbase_config/sample_ycsb_config.xml

./run_experiments.py ycsb_playground_readratio
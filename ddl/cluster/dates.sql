/*
 * Cluster lineitem and orders by dates.
 */

CLUSTER orders using o_od;
CREATE INDEX l_maxdate ON lineitem (greatest(l_receiptdate, l_commitdate));
CLUSTER lineitem using l_maxdate;
DROP INDEX l_maxdate;

/*
 * Cluster lineitem and orders by dates.
 */

CREATE INDEX o_od_temp ON orders (o_orderdate);
CLUSTER orders using o_od_temp;
drop index o_od_temp;
ANALYZE orders;

CREATE INDEX l_maxdate ON lineitem (greatest(l_receiptdate, l_commitdate));
CLUSTER lineitem using l_maxdate;
DROP INDEX l_maxdate;
ANALYZE lineitem;
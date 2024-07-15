/*
 * Cluster lineitem and orders by their primary key.
 */

CLUSTER orders using o_ok;
ANALYZE orders;

CLUSTER lineitem using l_ok_ln;
ANALYZE lineitem;
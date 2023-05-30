/*
 * BRIN indexes on all tables where it makes sense to use them
 * note: some minmax BRIN indexes, some bloom indexes where only equality is needed and 
 */


-- REGION and NATION are too small for BRIN to make sense

-- PART
DROP INDEX IF EXISTS p_bloom_pk;
DROP INDEX IF EXISTS p_bloom_size;
DROP INDEX IF EXISTS p_bloom_container;


-- SUPPLIER
DROP INDEX IF EXISTS s_bloom_sk;


-- PARTSUPP
DROP INDEX IF EXISTS ps_bloom_sc;
DROP INDEX IF EXISTS ps_bloom_pk;
DROP INDEX IF EXISTS ps_bloom_sk;
DROP INDEX IF EXISTS ps_bloom_pk_sk;


-- CUSTOMER
DROP INDEX IF EXISTS c_bloom_ck;
DROP INDEX IF EXISTS c_bloom_ms;
DROP INDEX IF EXISTS c_brin_ab;


-- ORDERS
-- note: already clustered by orderdate!
DROP INDEX IF EXISTS o_brin_od;
DROP INDEX IF EXISTS o_bloom_ok;
DROP INDEX IF EXISTS o_bloom_ck;


-- LINEITEM
DROP INDEX IF EXISTS l_brin_sd;
DROP INDEX IF EXISTS l_brin_cd;
DROP INDEX IF EXISTS l_brin_rd;
DROP INDEX IF EXISTS l_brin_discount;
DROP INDEX IF EXISTS l_brin_qty;
DROP INDEX IF EXISTS l_bloom_ok;
DROP INDEX IF EXISTS l_bloom_pk;
DROP INDEX IF EXISTS l_bloom_sk;
DROP INDEX IF EXISTS l_bloom_pk_sk;
DROP INDEX IF EXISTS l_bloom_sm;
DROP INDEX IF EXISTS l_bloom_si;


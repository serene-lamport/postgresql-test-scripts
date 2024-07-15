/*
 * Drop B-Tree indices and constraints created by btrees/create.sql
 */

-- REGION indices
ALTER TABLE region DROP CONSTRAINT IF EXISTS r_rk CASCADE;

-- NATION indices and constraints
ALTER TABLE nation DROP CONSTRAINT IF EXISTS n_nk CASCADE;
ALTER TABLE nation DROP CONSTRAINT IF EXISTS nation_n_regionkey_fkey CASCADE;
DROP INDEX IF EXISTS n_rk CASCADE;

-- PART indices
ALTER TABLE part DROP CONSTRAINT IF EXISTS p_pk CASCADE;

-- SUPPLIER indices and constraints
ALTER TABLE supplier DROP CONSTRAINT IF EXISTS s_sk CASCADE;
ALTER TABLE supplier DROP CONSTRAINT IF EXISTS supplier_s_nationkey_fkey CASCADE;
DROP INDEX IF EXISTS s_nk CASCADE;

-- PARTSUPP indices and constraints
ALTER TABLE partsupp DROP CONSTRAINT IF EXISTS ps_pk_sk CASCADE;
ALTER TABLE partsupp DROP CONSTRAINT IF EXISTS partsupp_ps_partkey_fkey CASCADE;
ALTER TABLE partsupp DROP CONSTRAINT IF EXISTS partsupp_ps_suppkey_fkey CASCADE;
DROP INDEX IF EXISTS ps_pk CASCADE;
DROP INDEX IF EXISTS ps_sk CASCADE;
DROP INDEX IF EXISTS ps_sk_pk CASCADE;

-- CUSTOMER indices and constraints
ALTER TABLE customer DROP CONSTRAINT IF EXISTS c_ck CASCADE;
ALTER TABLE customer DROP CONSTRAINT IF EXISTS customer_c_nationkey_fkey CASCADE;
DROP INDEX IF EXISTS c_nk CASCADE;

-- ORDERS indices and constraints
ALTER TABLE orders DROP CONSTRAINT IF EXISTS o_ok CASCADE;
ALTER TABLE orders DROP CONSTRAINT IF EXISTS orders_o_custkey_fkey CASCADE;
DROP INDEX IF EXISTS o_ck CASCADE;
DROP INDEX IF EXISTS o_od CASCADE;

-- LINEITEM indices and constraints
ALTER TABLE lineitem DROP CONSTRAINT IF EXISTS l_ok_ln CASCADE;
ALTER TABLE lineitem DROP CONSTRAINT IF EXISTS lineitem_l_orderkey_fkey CASCADE;
ALTER TABLE lineitem DROP CONSTRAINT IF EXISTS lineitem_l_partkey_l_suppkey_fkey CASCADE;
DROP INDEX IF EXISTS l_ok CASCADE;
DROP INDEX IF EXISTS l_pk CASCADE;
DROP INDEX IF EXISTS l_sk CASCADE;
DROP INDEX IF EXISTS l_sd CASCADE;
DROP INDEX IF EXISTS l_cd CASCADE;
DROP INDEX IF EXISTS l_rd CASCADE;
DROP INDEX IF EXISTS l_pk_sk CASCADE;
DROP INDEX IF EXISTS l_sk_pk CASCADE;

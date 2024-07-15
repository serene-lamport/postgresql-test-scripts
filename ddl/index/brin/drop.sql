/*
 * BRIN indexes on all tables where it makes sense to use them
 * note: some minmax BRIN indexes, some bloom indexes where only equality is needed and 
 */


-- REGION and NATION are too small for BRIN to make sense

-- PART
DROP INDEX IF EXISTS p_bloom_pk CASCADE;
DROP INDEX IF EXISTS p_bloom_size CASCADE;
DROP INDEX IF EXISTS p_bloom_container CASCADE;
DROP INDEX IF EXISTS p_bloom_brand CASCADE;

DROP INDEX IF EXISTS p_gin_name CASCADE;
DROP INDEX IF EXISTS p_gin_type CASCADE;
DROP INDEX IF EXISTS p_gin_brand CASCADE;


-- SUPPLIER
DROP INDEX IF EXISTS s_bloom_sk CASCADE;
DROP INDEX IF EXISTS s_bloom_nk CASCADE;

DROP INDEX IF EXISTS s_gin_comment CASCADE;


-- PARTSUPP
DROP INDEX IF EXISTS ps_bloom_sc CASCADE;
DROP INDEX IF EXISTS ps_bloom_pk CASCADE;
DROP INDEX IF EXISTS ps_brin_pk CASCADE;
DROP INDEX IF EXISTS ps_bloom_sk CASCADE;
DROP INDEX IF EXISTS ps_bloom_pk_sk CASCADE;


-- CUSTOMER
DROP INDEX IF EXISTS c_bloom_ck CASCADE;
DROP INDEX IF EXISTS c_bloom_ms CASCADE;
DROP INDEX IF EXISTS c_brin_ab CASCADE;
DROP INDEX IF EXISTS c_bloom_nk CASCADE;


-- ORDERS
-- note: already clustered by orderdate!
DROP INDEX IF EXISTS o_brin_od CASCADE;
DROP INDEX IF EXISTS o_bloom_ok CASCADE;
DROP INDEX IF EXISTS o_bloom_ck CASCADE;

DROP INDEX IF EXISTS o_gin_comment CASCADE;


-- LINEITEM
DROP INDEX IF EXISTS l_brin_sd CASCADE;
DROP INDEX IF EXISTS l_brin_cd CASCADE;
DROP INDEX IF EXISTS l_brin_rd CASCADE;
DROP INDEX IF EXISTS l_brin_discount CASCADE;
DROP INDEX IF EXISTS l_brin_qty CASCADE;
DROP INDEX IF EXISTS l_bloom_ok CASCADE;
DROP INDEX IF EXISTS l_bloom_pk CASCADE;
DROP INDEX IF EXISTS l_bloom_sk CASCADE;
DROP INDEX IF EXISTS l_bloom_pk_sk CASCADE;
DROP INDEX IF EXISTS l_bloom_sm CASCADE;
DROP INDEX IF EXISTS l_bloom_si CASCADE;


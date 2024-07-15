/*
 * BRIN indexes on all tables where it makes sense to use them
 * note: some minmax BRIN indexes, some bloom indexes where only equality is needed and 
 */


-- REGION and NATION are too small for BRIN to make sense

-- PART
CREATE INDEX p_bloom_pk on part USING BRIN (p_partkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX p_bloom_size on part USING BRIN (p_size int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX p_bloom_container on part USING BRIN (p_container bpchar_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX p_bloom_brand on part USING BRIN (p_brand bpchar_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);

CREATE INDEX p_gin_name ON part USING GIN(p_name gin_trgm_ops);
CREATE INDEX p_gin_type ON part USING GIN(p_type gin_trgm_ops);
-- CREATE INDEX p_gin_brand ON part USING GIN(p_brand gin_trgm_ops); -- doesn't support this datatype, use bloom instead

-- SUPPLIER
CREATE INDEX s_bloom_sk on supplier USING BRIN (s_suppkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX s_bloom_nk on supplier USING BRIN (s_nationkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);

CREATE INDEX s_gin_comment ON supplier USING GIN(s_comment gin_trgm_ops);



-- PARTSUPP
-- note: ordered by partkey
CREATE INDEX ps_bloom_sc on partsupp USING BRIN (ps_supplycost numeric_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX ps_brin_pk on partsupp USING BRIN (ps_partkey) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX ps_bloom_sk on partsupp USING BRIN (ps_suppkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
-- CREATE INDEX ps_bloom_pk_sk on partsupp USING BRIN (ps_partkey int4_bloom_ops, ps_suppkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
    -- row too large, but can just AND the other two indices anyways



-- CUSTOMER
CREATE INDEX c_bloom_ck ON customer USING BRIN (c_custkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX c_bloom_ms ON customer USING BRIN (c_mktsegment bpchar_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX c_brin_ab ON customer USING BRIN (c_acctbal) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX c_bloom_nk on customer USING BRIN (c_nationkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);


-- ORDERS
-- note: already clustered by orderdate!
CREATE INDEX o_brin_od ON orders USING BRIN (o_orderdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX o_bloom_ok ON orders USING BRIN (o_orderkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX o_bloom_ck ON orders USING BRIN (o_custkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);

CREATE INDEX o_gin_comment ON orders USING GIN(o_comment gin_trgm_ops);


-- LINEITEM
CREATE INDEX l_brin_sd on lineitem USING BRIN (l_shipdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_cd on lineitem USING BRIN (l_commitdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_rd on lineitem USING BRIN (l_receiptdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_discount on lineitem USING BRIN (l_discount) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_qty on lineitem USING BRIN (l_quantity) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_bloom_ok ON lineitem USING BRIN (l_orderkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX l_bloom_pk ON lineitem USING BRIN (l_partkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX l_bloom_sk ON lineitem USING BRIN (l_suppkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
-- CREATE INDEX l_bloom_pk_sk ON lineitem USING BRIN (l_partkey int4_bloom_ops, l_suppkey int4_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX l_bloom_sm ON lineitem USING BRIN (l_shipmode bpchar_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);
CREATE INDEX l_bloom_si ON lineitem USING BRIN (l_shipinstruct bpchar_bloom_ops) WITH (pages_per_range = REPLACEME_BLOOM_PAGES_PER_RANGE);


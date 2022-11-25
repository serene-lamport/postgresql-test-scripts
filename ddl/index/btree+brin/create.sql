/*
 * Create B-Tree and BRIN indices on all the TPCH tables.
 */

-- TODO: for now we only have BRIN indexes on linitem and only the ones relevant to Q1/Q6 for the microbenchmarks. Should add it for others as well

---------- INDICES: (based on looking at specification only)
    -- probably doesn't make sense to use BRIN for primary keys...
-- REGION: primary key, but only 5 rows so who cares
-- NATION: primary key and FK to region, but only 25 rows
-- PART: only primary key.
    -- ...
-- SUPPLIER: PK and FK to nation
    -- bloom filter on nation key?
-- PARTSUPP: (part, supplier) and (supplier, part) + FK on supplier and part separately
    -- ?? make sure to cluster appropriately...
-- CUSTOMER: PK and FK to nation
    -- bloom filter on nation key?
-- ORDERS: PK, FK to customers, INDEX on order date
    -- bloom on customer?
    -- BRIN on order date?
    -- status may also be correlated with date, check how data is generated...
-- LINEITEM: PK (order, linenumber), FK to orders, FK to partsupp, INDEX on part, supplier, all the dates
    -- BRIN on order (assuming sequential...)
    -- BRIN on all the dates
    -- Bloom for part and supplier?
    -- Bloom for partsupp maybe?


---------- PREDICATES: of things that might make sense to use BRIN, NOT including primary/foreign key =
-- REGION:
-- NATION:
-- PART:
    -- p_size = (Q2) in (Q19) --- only values between 0 and 50, maybe bloom filter here?
    -- p_type like '%...' (Q2) = (Q8) not like '...%' (Q16)
    -- p_name like '%...%' (Q8)
    -- p_brand <> (Q16) = (Q17, Q19)
    -- p_container = (Q17) in (Q19)
-- SUPPLIER:
-- PARTSUPP:
    -- ps_partkey = (Q2)
    -- ps_suppkey = (Q2) not in (Q16)
    -- ps_supplycost = [min value for other predicates] (Q2)
-- CUSTOMER:
    -- c_mktsegment = (Q3)
    -- c_acctbal > (Q22)
-- ORDERS:
    -- o_orderdate < (Q3); >= & < between (Q4, Q5, Q8, Q10)
    -- o_orderkey in (subquery) (Q18)
    -- o_orderstatus = (Q21)
-- LINEITEM:
    -- l_shipdate <= (Q1); > (Q3); >= & < between (Q6, Q7, Q14, Q15)
    -- l_commitdate < l_receiptdate (Q4, Q12, Q21)
    -- l_shipdate < l_commitdate (Q12)
    -- l_discount betweem (Q6)
    -- l_quantity < (Q6, Q17) between (Q19)
    -- l_shipmode in (Q12)
    -- l_receiptdate >= & < between (Q12)
    -- l_shipmode in (Q19)
    -- l_shipinstruct = (Q19)
    -- Orginal microbench: l_shipdate between {1%, 10%, 50%, 100%}, also l_discount and l_quantity for Q6? (use BRIN for all of these?)


---------- DATA GENERATION: things that might make sense to use BRIN based on data
-- REGION:
-- NATION:
-- PART:
    -- p_brand: bloom filter?
    -- p_container: bloom filter? (= and in)
-- SUPPLIER:
-- PARTSUPP:
-- CUSTOMER:
    -- c_mktsegment: bloom filter?
-- ORDERS:
    -- UNFORTUNATELY, dates not correlated with orderkey...
    -- o_orderdate: unfortunately random, maybe cluster by this?
    -- would like to use BRIN for dates.
-- LINEITEM:
    -- dates are correlated with orderdate BUT not generated in this order... is there a way to cluster by the order date? (can de-normalize and include orderdate...)
        -- update lineitem set l_orderdate = (select o_orderdate from orders where o_orderkey = l_orderkey); -- for sf=1: took 9m33 with indexes  (on tem112)
        -- update lineitem set l_orderdate = o_orderdate from orders where l_orderkey = o_orderkey; -- for sf=1: took 2m12 with no index, 8m22 with...? (on tem112)
    -- l_shipinstruct: bloom filter?
    -- l_shipmode: bloom filter?


----- HOW TO CLUSTER BY MAX DATE: create index l_maxdate on lineitem (greatest(l_commitdate, l_shipmode)); cluster linitem using l_maxdate; drop index l_maxdate;


--            count (sf=20)   size (sf=20)   size (sf=10)
-- region:                5         8 kB           -
-- nation:               25         8 kB           -
-- part:          4,000,000       640 MB         320 MB
-- supplier:        200,000        35 MB          17 MB
-- partsupp:     16,000,000       2.6 GB         1.3 GB
-- customer:      3,000,000       560 MB         280 MB
-- orders:       30,000,000         4 GB           2 GB
-- lineitem:    120,000,000        17 GB         8.6 GB
-- total size: 25 GiB (for SF=20)



-- REGION indices (5)
CREATE UNIQUE INDEX r_rk ON region (r_regionkey ASC);
ALTER TABLE region ADD PRIMARY KEY USING INDEX r_rk;

-- NATION indices and constraints (25)
CREATE UNIQUE INDEX n_nk ON nation (n_nationkey ASC);
ALTER TABLE nation ADD PRIMARY KEY USING INDEX n_nk;
CREATE INDEX n_rk ON nation (n_regionkey ASC);
ALTER TABLE nation ADD FOREIGN KEY (n_regionkey) REFERENCES region (r_regionkey) ON DELETE CASCADE;


-- PART indices (SF * 200,000)
CREATE UNIQUE INDEX p_pk ON part (p_partkey ASC);
ALTER TABLE part ADD PRIMARY KEY USING INDEX p_pk;

-- SUPPLIER indices and constraints (SF * 10,000)
CREATE UNIQUE INDEX s_sk ON supplier (s_suppkey ASC);
ALTER TABLE supplier ADD PRIMARY KEY USING INDEX s_sk;

CREATE INDEX s_nk ON supplier (s_nationkey ASC);
ALTER TABLE supplier ADD FOREIGN KEY (s_nationkey) REFERENCES nation (n_nationkey) ON DELETE CASCADE;

-- PARTSUPP indices and constraints (SF * 800,000)
CREATE INDEX ps_pk ON partsupp (ps_partkey ASC);
CREATE INDEX ps_sk ON partsupp (ps_suppkey ASC);
CREATE UNIQUE INDEX ps_pk_sk ON partsupp (ps_partkey ASC, ps_suppkey ASC);
ALTER TABLE partsupp ADD PRIMARY KEY USING INDEX ps_pk_sk;
CREATE UNIQUE INDEX ps_sk_pk ON partsupp (ps_suppkey ASC, ps_partkey ASC);

ALTER TABLE partsupp ADD FOREIGN KEY (ps_partkey) REFERENCES part (p_partkey) ON DELETE CASCADE;
ALTER TABLE partsupp ADD FOREIGN KEY (ps_suppkey) REFERENCES supplier (s_suppkey) ON DELETE CASCADE;

-- CUSTOMER indices and constraints (SF * 150,000)
CREATE UNIQUE INDEX c_ck ON customer (c_custkey ASC);
ALTER TABLE customer ADD PRIMARY KEY USING INDEX c_ck;
CREATE INDEX c_nk ON customer (c_nationkey ASC);
ALTER TABLE customer ADD FOREIGN KEY (c_nationkey) REFERENCES nation (n_nationkey) ON DELETE CASCADE;

-- ORDERS indices and constraints (SF * 1,500,000)
CREATE UNIQUE INDEX o_ok ON orders (o_orderkey ASC);
ALTER TABLE orders ADD PRIMARY KEY USING INDEX o_ok;
CREATE INDEX o_ck ON orders (o_custkey ASC);
CREATE INDEX o_od ON orders (o_orderdate ASC);
ALTER TABLE orders ADD FOREIGN KEY (o_custkey) REFERENCES customer (c_custkey) ON DELETE CASCADE;

CREATE INDEX o_brin_od ON orders USING BRIN (o_orderdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);

-- LINEITEM indices and constraints (~ SF * 600,000)
CREATE UNIQUE INDEX l_ok_ln ON lineitem (l_orderkey, l_linenumber);
ALTER TABLE lineitem ADD PRIMARY KEY USING INDEX l_ok_ln;
CREATE INDEX l_ok ON lineitem (l_orderkey ASC);
CREATE INDEX l_pk ON lineitem (l_partkey ASC);
CREATE INDEX l_sk ON lineitem (l_suppkey ASC);
CREATE INDEX l_sd ON lineitem (l_shipdate ASC);
CREATE INDEX l_cd ON lineitem (l_commitdate ASC);
CREATE INDEX l_rd ON lineitem (l_receiptdate ASC);
CREATE INDEX l_pk_sk ON lineitem (l_partkey ASC, l_suppkey ASC);
CREATE INDEX l_sk_pk ON lineitem (l_suppkey ASC, l_partkey ASC);

CREATE INDEX l_brin_sd on lineitem USING BRIN (l_shipdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_cd on lineitem USING BRIN (l_commitdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_rd on lineitem USING BRIN (l_receiptdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_discount on lineitem USING BRIN (l_discount) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_qty on lineitem USING BRIN (l_quantity) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);


ALTER TABLE lineitem ADD FOREIGN KEY (l_orderkey) REFERENCES orders (o_orderkey) ON DELETE CASCADE;
ALTER TABLE lineitem ADD FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey) ON DELETE CASCADE;



-- CREATE INDEX <index_name> ON <table> USING BRIN (<columns...>) [ WITH (pages_per_range = <num>) ];
-- CLUSTER <table> USING <index_name>;
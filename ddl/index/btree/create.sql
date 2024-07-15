/*
 * Create B-Tree indices on all the TPCH tables.
 */


-- REGION indices
CREATE UNIQUE INDEX r_rk ON region (r_regionkey ASC);
ALTER TABLE region ADD PRIMARY KEY USING INDEX r_rk;

-- NATION indices and constraints
CREATE UNIQUE INDEX n_nk ON nation (n_nationkey ASC);
ALTER TABLE nation ADD PRIMARY KEY USING INDEX n_nk;
CREATE INDEX n_rk ON nation (n_regionkey ASC);
ALTER TABLE nation ADD FOREIGN KEY (n_regionkey) REFERENCES region (r_regionkey) ON DELETE CASCADE;


-- PART indices
CREATE UNIQUE INDEX p_pk ON part (p_partkey ASC);
ALTER TABLE part ADD PRIMARY KEY USING INDEX p_pk;

-- SUPPLIER indices and constraints
CREATE UNIQUE INDEX s_sk ON supplier (s_suppkey ASC);
ALTER TABLE supplier ADD PRIMARY KEY USING INDEX s_sk;

CREATE INDEX s_nk ON supplier (s_nationkey ASC);
ALTER TABLE supplier ADD FOREIGN KEY (s_nationkey) REFERENCES nation (n_nationkey) ON DELETE CASCADE;

-- PARTSUPP indices and constraints
CREATE INDEX ps_pk ON partsupp (ps_partkey ASC);
CREATE INDEX ps_sk ON partsupp (ps_suppkey ASC);
CREATE UNIQUE INDEX ps_pk_sk ON partsupp (ps_partkey ASC, ps_suppkey ASC);
ALTER TABLE partsupp ADD PRIMARY KEY USING INDEX ps_pk_sk;
CREATE UNIQUE INDEX ps_sk_pk ON partsupp (ps_suppkey ASC, ps_partkey ASC);

ALTER TABLE partsupp ADD FOREIGN KEY (ps_partkey) REFERENCES part (p_partkey) ON DELETE CASCADE;
ALTER TABLE partsupp ADD FOREIGN KEY (ps_suppkey) REFERENCES supplier (s_suppkey) ON DELETE CASCADE;

-- CUSTOMER indices and constraints
CREATE UNIQUE INDEX c_ck ON customer (c_custkey ASC);
ALTER TABLE customer ADD PRIMARY KEY USING INDEX c_ck;
CREATE INDEX c_nk ON customer (c_nationkey ASC);
ALTER TABLE customer ADD FOREIGN KEY (c_nationkey) REFERENCES nation (n_nationkey) ON DELETE CASCADE;

-- ORDERS indices and constraints

CREATE UNIQUE INDEX o_ok ON orders (o_orderkey ASC);
ALTER TABLE orders ADD PRIMARY KEY USING INDEX o_ok;
CREATE INDEX o_ck ON orders (o_custkey ASC);
CREATE INDEX o_od ON orders (o_orderdate ASC);
ALTER TABLE orders ADD FOREIGN KEY (o_custkey) REFERENCES customer (c_custkey) ON DELETE CASCADE;

-- LINEITEM indices and constraints
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


ALTER TABLE lineitem ADD FOREIGN KEY (l_orderkey) REFERENCES orders (o_orderkey) ON DELETE CASCADE;
ALTER TABLE lineitem ADD FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey) ON DELETE CASCADE;



-- CREATE INDEX <index_name> ON <table> USING BRIN (<columns...>) [ WITH (pages_per_range = <num>) ];
-- CLUSTER <table> USING <index_name>;
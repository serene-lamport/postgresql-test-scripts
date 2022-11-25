/*
 * Only the BRIN indexes on lineitem that are relevant for the microbenchmarks.
 */


CREATE INDEX l_brin_sd on lineitem USING BRIN (l_shipdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_cd on lineitem USING BRIN (l_commitdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_rd on lineitem USING BRIN (l_receiptdate) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_discount on lineitem USING BRIN (l_discount) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
CREATE INDEX l_brin_qty on lineitem USING BRIN (l_quantity) WITH (pages_per_range = REPLACEME_BRIN_PAGES_PER_RANGE);
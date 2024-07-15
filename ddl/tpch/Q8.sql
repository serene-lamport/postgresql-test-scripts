SELECT
   o_year,
   SUM(
   CASE
	  WHEN
		 nation = 'CANADA' --- nation
	  THEN
		 volume
	  ELSE
		 0
   END
) / SUM(volume) AS mkt_share
FROM
   (
	  SELECT
		 EXTRACT(YEAR
	  FROM
		 o_orderdate) AS o_year,
		 l_extendedprice * (1 - l_discount) AS volume,
		 n2.n_name AS nation
	  FROM
		 part,
		 supplier,
		 lineitem,
		 orders,
		 customer,
		 nation n1,
		 nation n2,
		 region
	  WHERE
		 p_partkey = l_partkey
		 AND s_suppkey = l_suppkey
		 AND l_orderkey = o_orderkey
		 AND o_custkey = c_custkey
		 AND c_nationkey = n1.n_nationkey
		 AND n1.n_regionkey = r_regionkey
		 AND r_name = 'AMERICA' --- region corresponding to nation above
		 AND s_nationkey = n2.n_nationkey
		 AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
		 AND p_type = 'STANDARD PLATED COPPER' --- ("STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO") ("ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED") ("TIN", "NICKEL", "BRASS", "STEEL", "COPPER")
   )
   AS all_nations
GROUP BY
   o_year
ORDER BY
   o_year
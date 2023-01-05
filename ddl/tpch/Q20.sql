-- explain (costs, verbose, format json)
SELECT
   s_name,
   s_address
FROM
   supplier,
   nation
WHERE
   s_suppkey IN
   (
	  SELECT
		 ps_suppkey
	  FROM
		 partsupp
	  WHERE
		 ps_partkey IN
		 (
			SELECT
			   p_partkey
			FROM
			   part
			WHERE
			   p_name LIKE 'blue%' -- see name generator
		 )
		 AND ps_availqty > (
		 SELECT
			0.5 * SUM(l_quantity)
		 FROM
			lineitem
		 WHERE
			l_partkey = ps_partkey
			AND l_suppkey = ps_suppkey
			AND l_shipdate >= DATE '1995-01-01' --- 1993-1997
			AND l_shipdate < DATE '1995-01-01' + INTERVAL '1' YEAR ) --- SAME date
   )
   AND s_nationkey = n_nationkey
   AND n_name = 'CANADA' --- nation
ORDER BY
   s_name
;
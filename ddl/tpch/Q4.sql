SELECT
   o_orderpriority,
   COUNT(*) AS order_count
FROM
   orders
WHERE
   o_orderdate >= DATE '1995-01-01' --- year between 1993 and 1997, month between 1 to 10
   AND o_orderdate < DATE '1995-01-01' + INTERVAL '3' MONTH --- SAME date
   AND EXISTS
   (
	  SELECT
		 *
	  FROM
		 lineitem
	  WHERE
		 l_orderkey = o_orderkey
		 AND l_commitdate < l_receiptdate
   )
GROUP BY
   o_orderpriority
ORDER BY
   o_orderpriority
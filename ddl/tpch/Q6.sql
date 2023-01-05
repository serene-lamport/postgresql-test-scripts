SELECT
   SUM(l_extendedprice * l_discount) AS revenue
FROM
   lineitem
WHERE
   l_shipdate >= DATE '1995-01-01' --- 1993 to 1997
   AND l_shipdate < DATE '1995-01-01' + INTERVAL '1' YEAR --- SAME date
   AND l_discount BETWEEN 0.05 - 0.01 AND 0.05 + 0.01 --- discount between 0.02 and 0.09 (same value twice +- 0.01)
   AND l_quantity < 24 --- 24 or 25
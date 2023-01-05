SELECT
   l_orderkey,
   SUM(l_extendedprice * (1 - l_discount)) AS revenue,
   o_orderdate,
   o_shippriority
FROM
   customer,
   orders,
   lineitem
WHERE
   c_mktsegment = 'AUTOMOBILE' --- "AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"
   AND c_custkey = o_custkey
   AND l_orderkey = o_orderkey
   AND o_orderdate < DATE '1995-03-01' --- [1995-03-01, 1995-03-31]
   AND l_shipdate > DATE '1995-03-01' --- SAME date
GROUP BY
   l_orderkey,
   o_orderdate,
   o_shippriority
ORDER BY
   revenue DESC,
   o_orderdate LIMIT 10
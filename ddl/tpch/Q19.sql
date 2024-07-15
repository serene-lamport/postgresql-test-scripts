SELECT
   SUM(l_extendedprice* (1 - l_discount)) AS revenue
FROM
   lineitem,
   part
WHERE
   (
	  p_partkey = l_partkey
	  AND p_brand = 'Brand#42' --- Brand#?? (digits 1-5)
	  AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
	  AND l_quantity >= 7 --- 1-10
	  AND l_quantity <= 7 + 10 --- SAME quantity
	  AND p_size BETWEEN 1 AND 5
	  AND l_shipmode IN ('AIR', 'AIR REG')
	  AND l_shipinstruct = 'DELIVER IN PERSON'
   )
   OR
   (
	  p_partkey = l_partkey
	  AND p_brand = 'Brand#27' --- Brand#?? (any digits 1-5, not the same as before)
	  AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
	  AND l_quantity >= 10 -- 10-20
	  AND l_quantity <= 10 + 10 --- SAME quantity
	  AND p_size BETWEEN 1 AND 10
	  AND l_shipmode IN ('AIR', 'AIR REG')
	  AND l_shipinstruct = 'DELIVER IN PERSON'
   )
   OR
   (
	  p_partkey = l_partkey
	  AND p_brand = 'Brand#11' --- Brand#?? (digits 1-5)
	  AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
	  AND l_quantity >= 30 --- 20-30
	  AND l_quantity <= 30 + 10 --- SAME quantity
	  AND p_size BETWEEN 1 AND 15
	  AND l_shipmode IN ('AIR', 'AIR REG')
	  AND l_shipinstruct = 'DELIVER IN PERSON'
   )
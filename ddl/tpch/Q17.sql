SELECT
   SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM
   lineitem,
   part
WHERE
   p_partkey = l_partkey
   AND p_brand = 'Brand#42' --- 'Brand#??' with 2 digits in 1-5
   AND p_container = 'LG BOX' --- ("SM", "LG", "MED", "JUMBO", "WRAP") ("CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM")
   AND l_quantity < (
   SELECT
	  0.2 * AVG(l_quantity)
   FROM
	  lineitem
   WHERE
	  l_partkey = p_partkey )
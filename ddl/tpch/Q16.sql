SELECT
   p_brand,
   p_type,
   p_size,
   COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM
   partsupp,
   part
WHERE
   p_partkey = ps_partkey
   AND p_brand <> 'Brand#42' --- "Brand#??" -- 2 digits 1-5
   AND p_type NOT LIKE 'SMALL BRUSHED%' --- ("STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO") ("ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED")%
   AND p_size IN (1, 7, 12, 24, 36, 42, 49, 50) --- 8 different values in [1, 50]
   AND ps_suppkey NOT IN
   (
	  SELECT
		 s_suppkey
	  FROM
		 supplier
	  WHERE
		 s_comment LIKE '%Customer%Complaints%'
   )
GROUP BY
   p_brand,
   p_type,
   p_size
ORDER BY
   supplier_cnt DESC,
   p_brand,
   p_type,
   p_size
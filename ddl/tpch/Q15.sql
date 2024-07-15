-- DROP VIEW IF EXISTS revenue0;
-- CREATE view revenue0 (supplier_no, total_revenue) AS
-- 	SELECT
-- 	   l_suppkey,
-- 	   SUM(l_extendedprice * (1 - l_discount))
-- 	FROM
-- 	   lineitem
-- 	WHERE
-- 	   l_shipdate >= DATE '1995-04-01' --- 1993-1997, month restricted in 1997
-- 	   AND l_shipdate < DATE '1995-04-01' + INTERVAL '3' MONTH --- SAME date
-- 	GROUP BY
-- 	   l_suppkey;
--
-- SELECT
--    s_suppkey,
--    s_name,
--    s_address,
--    s_phone,
--    total_revenue
-- FROM
--    supplier,
--    revenue0
-- WHERE
--    s_suppkey = supplier_no
--    AND total_revenue = (
-- 	  SELECT
-- 		 MAX(total_revenue)
-- 	  FROM
-- 		 revenue0
--    )
-- ORDER BY
--    s_suppkey;
--
-- drop view revenue0; --- note: check "data output" tab for the actual results!


WITH revenue0 (supplier_no, total_revenue) AS (
    SELECT
        l_suppkey,
        SUM(l_extendedprice * (1 - l_discount))
    FROM
        lineitem
    WHERE
        l_shipdate >= DATE '1995-04-01' --- 1993-1997, month restricted in 1997
        AND l_shipdate < DATE '1995-04-01' + INTERVAL '3' MONTH --- SAME date
    GROUP BY
        l_suppkey
)
SELECT
    s_suppkey,
    s_name,
    s_address,
    s_phone,
    total_revenue
FROM
    supplier,
    revenue0
WHERE
    s_suppkey = supplier_no
    AND total_revenue = (
        SELECT
        MAX(total_revenue)
        FROM
        revenue0
    )
ORDER BY
    s_suppkey;
SELECT
   ps_partkey,
   SUM(ps_supplycost * ps_availqty) AS VALUE
FROM
   partsupp,
   supplier,
   nation
WHERE
   ps_suppkey = s_suppkey
   AND s_nationkey = n_nationkey
   AND n_name = 'ETHIOPIA'
GROUP BY
   ps_partkey
HAVING
   SUM(ps_supplycost * ps_availqty) > (
   SELECT
	  SUM(ps_supplycost * ps_availqty) * 0.001 --- 0.0001 / scale factor --- TODO!!!! change this in the code!
   FROM
	  partsupp, supplier, nation
   WHERE
	  ps_suppkey = s_suppkey
	  AND s_nationkey = n_nationkey
	  AND n_name = 'CANADA' ) --- nation
   ORDER BY
	  VALUE DESC
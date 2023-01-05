SELECT
   cntrycode,
   COUNT(*) AS numcust,
   SUM(c_acctbal) AS totacctbal
FROM
   (
	  SELECT
		 SUBSTRING(c_phone FROM 1 FOR 2) AS cntrycode,
		 c_acctbal
	  FROM
		 customer
	  WHERE
		 SUBSTRING(c_phone FROM 1 FOR 2) IN ('10', '12', '17', '27', '28', '32', '34') --- 7 distinct numbers in 10-34 (but strings)
		 AND c_acctbal > 
		 (
			 SELECT
				AVG(c_acctbal)
			 FROM
				customer
			 WHERE
				c_acctbal > 0.00
				AND SUBSTRING(c_phone FROM 1 FOR 2) IN ('10', '12', '17', '27', '28', '32', '34') -- the SAME 7 numbers
		 )
		 AND NOT EXISTS
		 (
			 SELECT
				*
			 FROM
				orders
			 WHERE
				o_custkey = c_custkey
		 )
   )
   AS custsale
GROUP BY
   cntrycode
ORDER BY
   cntrycode
/*
 * Drop indices and constraints created by lineitem_brinonly/create.sql
 */

DROP INDEX IF EXISTS l_brin_sd CASCADE;
DROP INDEX IF EXISTS l_brin_cd CASCADE;
DROP INDEX IF EXISTS l_brin_rd CASCADE;
DROP INDEX IF EXISTS l_brin_discount CASCADE;
DROP INDEX IF EXISTS l_brin_qty CASCADE;

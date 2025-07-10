DROP TABLE IF EXISTS usertable;
CREATE TABLE usertable (
    ycsb_key int PRIMARY KEY,
    seqscan_key int, -- To add sequential scans to ycsb by clustering the table based on this column and adding an index of type brin
    field1   text,
    field2   text,
    field3   text,
    field4   text,
    field5   text,
    field6   text,
    field7   text,
    field8   text,
    field9   text,
    field10  text
);

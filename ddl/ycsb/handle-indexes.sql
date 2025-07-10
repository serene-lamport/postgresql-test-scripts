

create index tempidx on usertable(seqscan_key); 

cluster usertable using tempidx;

drop index tempidx;

create index idx_seqscan_key on usertable using brin(seqscan_key);

analyze usertable;

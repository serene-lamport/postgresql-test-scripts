TODO
====

- [ ] revisit clustering: is it better to simply cluster every time?
- [ ] index defs:
    - [ ] add BRIN/bloom for other tables
    - [ ] add a brin-mostly setup (only brin/blook for ones where that makes sense, keep the btrees where they can't be replaced)
    - [ ] brin only with no btrees (at least for the large tables)
- [ ] process results:
    - [ ] ...?
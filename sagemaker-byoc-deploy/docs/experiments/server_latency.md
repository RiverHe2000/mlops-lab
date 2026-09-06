| Batch rows | Content type | p50 ms | p95 ms | p99 ms | rows / s |
|---:|---|---:|---:|---:|---:|
| 1 | text/csv | 7.23 | 7.57 | 8.24 | 138 |
| 1 | application/json | 7.16 | 7.73 | 10.05 | 140 |
| 10 | text/csv | 7.42 | 7.99 | 11.56 | 1,348 |
| 10 | application/json | 7.42 | 8.10 | 9.30 | 1,348 |
| 100 | text/csv | 7.62 | 8.31 | 8.31 | 13,123 |
| 100 | application/json | 7.69 | 8.12 | 8.12 | 12,997 |
| 1000 | text/csv | 10.10 | 10.83 | 10.83 | 99,039 |
| 1000 | application/json | 11.11 | 11.98 | 11.98 | 90,002 |

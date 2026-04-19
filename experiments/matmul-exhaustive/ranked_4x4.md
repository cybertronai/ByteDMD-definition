| rank | strategy | cost | peak_scratch | peak_arg | n_reads |
|-----:|----------|-----:|-------------:|---------:|--------:|
| 1 | `naive_b_col_cached` | 739 | 21 | 32 | 256 |
| 2 | `outer_b_row_a_scalar` | 741 | 22 | 32 | 272 |
| 3 | `naive_a_row_scalar_acc` | 801 | 22 | 32 | 272 |
| 4 | `naive_ijk_direct` | 812 | 17 | 32 | 240 |
| 5 | `naive_jik_direct` | 812 | 17 | 32 | 240 |
| 6 | `naive_kij_rank1` | 812 | 17 | 32 | 240 |
| 7 | `block_2x2_direct` | 847 | 21 | 32 | 240 |
| 8 | `batched_per_pair` | 849 | 20 | 32 | 240 |
| 9 | `naive_a_row_cached` | 850 | 21 | 32 | 256 |
| 10 | `naive_ijk_always_acc` | 882 | 17 | 32 | 272 |
| 11 | `batched_all_64` | 1352 | 80 | 32 | 240 |

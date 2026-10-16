# Cleaning log v1

Run 2026-10-13T06:26:52Z by code/clean_v1.py; raw checksums 9/9 OK. Rules in the script docstring.

## Rows remaining after each rule

```
type                                  lake           node           reach           total
v                                        C      D       C       D       C       D        
raw                                  39378  45207  132748  168899  125631  121678  633541
R1 drop BOR gauges                   33070  37699  132748  168899  125631  121678  619725
R1 drop |dt| > 1 h                   33070  37699  132748  168899  125631  121678  619725
R2a conflict: keep PGC0 rows         32953  37699  132748  168899  125631  121678  619608
R2b conflict unresolved: drop group  32919  37699  132680  168899  125555  121678  619430
R2c collapse repeated SWOT obs       29700  31698  125640  160476  119752  116133  583399
```

## Flags (rows flagged; rows are kept)

```
         calval  tidal_gauge  mixed_datum_site  shared_swot_obs  egm_unstable
type  v                                                                      
lake  C    1628            0                 0             5379          1525
      D    1899            0                 0             5564          1528
node  C   14609         5322                 0                4          2391
      D   17976         6327                 0                0          2834
reach C   14011         5084                 0             9290          2274
      D   13518         4624                 0             5476          2036
```
gauges: tidal candidates 75, EGM offset unstable 52 of 1818

## Values set to NaN (R4)

{'area_tot_u >= 1e11': 7, 'area_det_u >= 1e11': 7, 'width < 0': 2, 'area_total < 0': 2, 'area_detct < 0': 4, 'area_wse < 0': 7}

## Label A coverage (rows by egm_source)

```
egm_source  native  node_transfer   none
type  v                                 
lake  C       5913              0  23787
      D      13548              0  18150
node  C      83672              0  41968
      D     116317              0  44159
reach C          0          87273  32479
      D          0          84609  31524
```

crosswalk key duplicates within version: C 108, D 70
C/D matched overpasses by type: {'lake': 21531, 'node': 88303, 'reach': 62169}

checks passed: unique overpass rows, pairing rule, eB centred per site, unique row_id

## Final table

```
           rows  sites  gauges  features  science_rows  share_|eB|>0.25m
type  v                                                                 
lake  C   29700    434     432       404         28072             0.305
      D   31698    439     437       409         29799             0.222
node  C  125640   2103    2103      2102        111031             0.443
      D  160476   2130    2130      2130        142500             0.436
reach C  119752   2099    2099      1987        105741             0.415
      D  116133   2110    2110      1999        102615             0.343
```

## Outputs

- data/processed/paired_clean_v1.parquet: 90450791 bytes, sha256 384b314fc130fabb2f34df0bb598345610a0a9eec1d424bef06a2806a321c3cc
- data/processed/cd_crosswalk_v1.parquet: 10268099 bytes, sha256 1e81abcf691b7b6c81d4da1bf7102253922e6c4ea94e714ebe2aceef5191071d

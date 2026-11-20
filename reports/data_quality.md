# Data Quality Report

_Generated 2026-11-17T06:47:03Z · config `d014eeff5f86`_

## Status: **PASS**

- Rows: **340**
- Errors: **0**
- Warnings: **2**

## Findings

| Severity | Code | Count | Message | Examples |
| --- | --- | --- | --- | --- |
| WARNING | gold_coverage | 0 | Rows usable as ground truth (human-annotated with a score) |  |
| WARNING | no_human_annotations | 0 | No human LMP annotations exist yet. The pipeline can be exercised end-to-end on draft labels, but NO experiment result may be reported as final until human annotation is complete. See src/annotation/annotation_app.py. |  |


## Distributions

### Language

| Language | Rows | Share |
| --- | --- | --- |
| en | 170 | 50.0% |
| hi | 170 | 50.0% |


### Legal domain

| Legal domain | Rows | Share |
| --- | --- | --- |
| contract | 202 | 59.4% |
| cyber | 72 | 21.2% |
| sale_of_goods | 66 | 19.4% |


### Split

| Split | Rows | Share |
| --- | --- | --- |
| train | 234 | 68.8% |
| test | 56 | 16.5% |
| validation | 50 | 14.7% |


### Annotation status

| Annotation status | Rows | Share |
| --- | --- | --- |
| draft | 340 | 100.0% |


### Language x domain grid

| language | contract | cyber | sale_of_goods |
| --- | --- | --- | --- |
| en | 101 | 36 | 33 |
| hi | 101 | 36 | 33 |


### Human LMP score distribution

**No human annotations yet.** Every score is either absent or a non-gold draft. No experiment may be reported as final. `TODO_HUMAN_ANNOTATION`


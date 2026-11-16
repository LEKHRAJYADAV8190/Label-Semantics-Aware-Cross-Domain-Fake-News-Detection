# Table 7 Extra 10-dataset DistilBERT (does not replace Tables 1-4)
Datasets used: ISOT, LIAR_PLUS, FakeVsSatire, Celebrity, Cidii, Fakes, Horne, Infodemic, NDF, Politifact
Locked R1-R10 unchanged. Cidii included only if processed CSV exists.

| run_id | phase | target | n_sources | n_train | n_test | accuracy | macro_f1 |
|---|---|---|---|---|---|---|---|
| X_in_Celebrity | in_domain | Celebrity | 1 | 300 | 100 | 0.820000 | 0.819712 |
| X_in_Cidii | in_domain | Cidii | 1 | 431 | 145 | 0.910345 | 0.906872 |
| X_in_Fakes | in_domain | Fakes | 1 | 469 | 156 | 0.532051 | 0.359325 |
| X_in_Horne | in_domain | Horne | 1 | 196 | 64 | 0.625000 | 0.384615 |
| X_in_Infodemic | in_domain | Infodemic | 1 | 6417 | 2140 | 0.974299 | 0.974225 |
| X_in_NDF | in_domain | NDF | 1 | 210 | 69 | 0.666667 | 0.400000 |
| X_in_Politifact | in_domain | Politifact | 1 | 570 | 190 | 0.847368 | 0.846137 |
| X_lodo_ISOT | lodo_9source | ISOT | 9 | 19135 | 900 | 0.684444 | 0.667705 |
| X_lodo_LIAR_PLUS | lodo_9source | LIAR_PLUS | 9 | 13127 | 1267 | 0.444357 | 0.354376 |
| X_lodo_FakeVsSatire | lodo_9source | FakeVsSatire | 9 | 23001 | 73 | 0.726027 | 0.710317 |
| X_lodo_Celebrity | lodo_9source | Celebrity | 9 | 23035 | 100 | 0.600000 | 0.554367 |
| X_lodo_Cidii | lodo_9source | Cidii | 9 | 22904 | 145 | 0.475862 | 0.425323 |
| X_lodo_Fakes | lodo_9source | Fakes | 9 | 22866 | 156 | 0.519231 | 0.393625 |
| X_lodo_Horne | lodo_9source | Horne | 9 | 23139 | 64 | 0.593750 | 0.558855 |
| X_lodo_Infodemic | lodo_9source | Infodemic | 9 | 16918 | 2140 | 0.744393 | 0.738805 |
| X_lodo_NDF | lodo_9source | NDF | 9 | 23125 | 69 | 0.666667 | 0.575328 |
| X_lodo_Politifact | lodo_9source | Politifact | 9 | 22765 | 190 | 0.615789 | 0.606625 |

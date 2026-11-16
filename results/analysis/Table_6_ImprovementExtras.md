# Table 6 Extra improvement runs

|run_id|target|method|accuracy|macro_f1|
|---|---|---|---|---|
|R4 locked|ISOT|LODO DistilBERT 2ep|0.6189|0.5867|
|E5 extra|ISOT|DistilBERT-class-weighted|0.6344|0.6265|
|E8 extra|ISOT|weighted 4ep max_len256|0.5622|0.5621|
|E13 extra|ISOT|LoRA-Random-K64-balanced|0.8778|0.8777|
|R5 locked|LIAR_PLUS|LODO DistilBERT 2ep|0.4538|0.3778|
|E6 extra|LIAR_PLUS|DistilBERT-class-weighted|0.4720|0.4155|
|E9 extra|LIAR_PLUS|weighted 4ep|0.4862|0.4565|
|E2 extra|LIAR_PLUS|LoRA-Random-K64-balanced|0.5328|0.5232|
|E11 extra|LIAR_PLUS|LoRA-Random-K128-balanced|0.5272|0.5259|
|R6 locked|FakeVsSatire|LODO DistilBERT 2ep|0.4247|0.3917|
|E7 extra|FakeVsSatire|DistilBERT-class-weighted|0.4384|0.4020|
|E10 extra|FakeVsSatire|weighted 4ep max_len256|0.3973|0.3172|
|E4 extra|FakeVsSatire|LoRA-Random-K64-balanced|0.5890|0.5871|
|E12 extra|FakeVsSatire|LoRA-Random-K128-balanced|0.6712|0.6605|

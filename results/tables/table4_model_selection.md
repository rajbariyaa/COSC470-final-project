**Learning-rate selection on validation data only**

| Method | Learning rate | Validation BCE (mean ± SD over seeds) | Selected |
|---|---|---|---|
| Centralized | 0.0003 | 0.3897 ± 0.0431 |  |
| Centralized | 0.001 | 0.3856 ± 0.0463 |  |
| Centralized | 0.003 | 0.3856 ± 0.0463 | yes |
| Centralized | 0.01 | 0.3898 ± 0.0461 |  |
| FedAvg | 0.0003 | 0.5169 ± 0.0320 |  |
| FedAvg | 0.001 | 0.4205 ± 0.0436 |  |
| FedAvg | 0.003 | 0.4151 ± 0.0453 |  |
| FedAvg | 0.01 | 0.4149 ± 0.0434 | yes |
| FedAvg with local fine-tuning | 3e-05 | 0.4110 ± 0.0430 |  |
| FedAvg with local fine-tuning | 0.0001 | 0.4037 ± 0.0426 |  |
| FedAvg with local fine-tuning | 0.0003 | 0.3913 ± 0.0424 |  |
| FedAvg with local fine-tuning | 0.001 | 0.3748 ± 0.0424 | yes |

**Test metrics, mean ± SD over 10 paired seeds (threshold 0.5)**

| Scope | Method | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|---|
| Cleveland | Centralized | 0.846 ± 0.048 | 0.842 ± 0.064 | 0.825 ± 0.106 | 0.828 ± 0.064 | 0.914 ± 0.024 |
| Cleveland | FedAvg | 0.849 ± 0.041 | 0.840 ± 0.067 | 0.839 ± 0.072 | 0.836 ± 0.044 | 0.910 ± 0.028 |
| Cleveland | FedAvg with local fine-tuning | 0.848 ± 0.048 | 0.848 ± 0.073 | 0.825 ± 0.096 | 0.831 ± 0.057 | 0.911 ± 0.029 |
| Hungary | Centralized | 0.824 ± 0.052 | 0.785 ± 0.067 | 0.717 ± 0.122 | 0.744 ± 0.087 | 0.895 ± 0.039 |
| Hungary | FedAvg | 0.834 ± 0.043 | 0.775 ± 0.071 | 0.782 ± 0.104 | 0.774 ± 0.060 | 0.896 ± 0.048 |
| Hungary | FedAvg with local fine-tuning | 0.834 ± 0.047 | 0.828 ± 0.078 | 0.695 ± 0.093 | 0.752 ± 0.073 | 0.902 ± 0.047 |
| Switzerland | Centralized | 0.912 ± 0.041 | 0.958 ± 0.034 | 0.949 ± 0.047 | 0.952 ± 0.023 | 0.820 ± 0.166 |
| Switzerland | FedAvg | 0.800 ± 0.082 | 0.964 ± 0.034 | 0.816 ± 0.080 | 0.882 ± 0.053 | 0.822 ± 0.154 |
| Switzerland | FedAvg with local fine-tuning | 0.932 ± 0.027 | 0.955 ± 0.035 | 0.975 ± 0.022 | 0.964 ± 0.014 | 0.833 ± 0.151 |
| VA Long Beach | Centralized | 0.725 ± 0.051 | 0.805 ± 0.033 | 0.829 ± 0.075 | 0.815 ± 0.039 | 0.718 ± 0.079 |
| VA Long Beach | FedAvg | 0.759 ± 0.050 | 0.815 ± 0.046 | 0.876 ± 0.045 | 0.843 ± 0.033 | 0.727 ± 0.082 |
| VA Long Beach | FedAvg with local fine-tuning | 0.767 ± 0.057 | 0.815 ± 0.040 | 0.886 ± 0.051 | 0.848 ± 0.040 | 0.727 ± 0.079 |
| Pooled | Centralized | 0.821 ± 0.027 | 0.845 ± 0.026 | 0.832 ± 0.059 | 0.837 ± 0.029 | 0.898 ± 0.019 |
| Pooled | FedAvg | 0.818 ± 0.021 | 0.841 ± 0.036 | 0.833 ± 0.049 | 0.835 ± 0.021 | 0.887 ± 0.022 |
| Pooled | FedAvg with local fine-tuning | 0.837 ± 0.029 | 0.857 ± 0.030 | 0.849 ± 0.046 | 0.852 ± 0.028 | 0.904 ± 0.021 |
| Equal-hospital mean | Centralized | 0.827 ± 0.021 | 0.848 ± 0.025 | 0.830 ± 0.063 | 0.835 ± 0.034 | 0.837 ± 0.043 |
| Equal-hospital mean | FedAvg | 0.811 ± 0.020 | 0.848 ± 0.033 | 0.828 ± 0.052 | 0.834 ± 0.022 | 0.839 ± 0.047 |
| Equal-hospital mean | FedAvg with local fine-tuning | 0.845 ± 0.026 | 0.861 ± 0.032 | 0.845 ± 0.048 | 0.849 ± 0.031 | 0.843 ± 0.045 |

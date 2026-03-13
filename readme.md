# Time series incident detection

## Experiments and conclusions

* **Feature engineering** Expanding raw data with window statistics (mean, standard deviation, trend slope) initially drastically improved the model's predictive capabilities.
* **Harder to predict data** Improving data generator process demonstrated that the basic sliding window without additional statistical features actually achieved better overall results. The model was able to extract a stronger, more reliable leading signal directly from the refined raw data than from artificially engineered metrics.
* **Optimizing** Optimizing the window width, horizon and amount of feature engineering led to expectable results that the model chose to only predict 1 step forward. Locking the horizon at 3 steps transformed a purely mathematical problem to a viable business solution, giving engineers actual time to react.
* **Algorithms** comparison. Comparing random forest to gradient boosting demonstrated the superiority of random forest in this specific environment. Built-in handling of imbalanced classes and high robustness to noise provided much more stable results
* **Switch** from F1 to F2. Shifting the optimization objective to the F2-Score prioritized incident detection over minimizing false alarms. Grid search reacted perfectly, lowering the decision threshold to 0.4, boosting recall to 85%
  
## Final architecture
| Component | Selected Value | Justification |
| --- | --- | --- |
| Algorithm | Random forest | Robust to noise and native handling of imbalanced incidents |
| Horizon | 3 steps | Ensures realistic recreation time for maintenance |
| Optimization target | F2 Metric | Consciously penalizes the model more heavily for missing critical failure |
| Alert Threshold | 0.4 | A direct result of favoring Recall over Precision, increasing the overall sensitivity of the system |
| Feature Engineering | Raw data | Experiments show that the latest raw values provide the most accurate signal |

## Final results
The final model was evaluated on a strictly isolated test set (the last 20% of the timeline) to prevent data leakage during hyperparameter tuning.

### Classification Report

                  precision    recall  f1-score   support

               0       0.97      0.92      0.95       852
               1       0.66      0.85      0.74       146

        accuracy                           0.91       998
       macro avg       0.82      0.89      0.85       998
    weighted avg       0.93      0.91      0.92       998

### Confusion Matrix

| | Predicted Normal (0) | Predicted Incident (1) |
|---|---|---|
| **Actual Normal (0)** | 788 (TN) | 64 (FP) |
| **Actual Incident (1)** | 22 (FN) | 124 (TP) |

### Top Feature Importances

With the Lookback Window set to 10 and raw data used as features, the Random Forest model relies heavily on the most recent time steps to make its predictions:

1. `x10`: 0.3787
2. `x9`: 0.1948
3. `x8`: 0.1214
4. `x7`: 0.0747
5. `x6`: 0.0540
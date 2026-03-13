import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_steps=2000, anomaly_prob=0.03, signal_length=10):
    np.random.seed(42)

    # base noise
    metrics = np.random.normal(loc=50, scale=5, size=n_steps)
    labels = np.zeros(n_steps)
    
    i = signal_length
    while i < n_steps:
        if np.random.rand() < anomaly_prob:
            
            # injecting a pattern before the incident
            for j in range(signal_length):
                metrics[i - signal_length + j] += (j * 2) 
                
            # incident
            metrics[i] += 40
            labels[i] = 1
            
            # skip a few steps so that incidents dont overlap
            i += signal_length 
        else:
            i += 1
            
    return metrics, labels

metrics, labels = generate_data()

plt.figure(figsize=(15, 5))
plt.plot(metrics, label="Data", color="#0080ff", alpha=0.8)

#anomaly_indices = np.where(labels == 1)[0]
#plt.scatter(anomaly_indices, metrics[anomaly_indices], color="red", label="incident", zorder=5)

plt.xlabel("time step")
plt.ylabel("metric value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def plot_predictions(metrics, y_test, y_pred, split_index, W):
    plt.figure(figsize=(15, 5))
    
    plt.plot(metrics, label="metric value", color="#0080ff", alpha=0.5)
    
    test_start_idx = split_index + W
    
    plt.axvline(x=test_start_idx, color='gray', linestyle='--', label="test set")
    
    # true positives, false positives and false negatives
    tp_indices = []
    fp_indices = []
    fn_indices = []
    
    for k in range(len(y_test)):
        actual = y_test[k]
        predicted = y_pred[k]
        plot_idx = split_index + k + W
        
        if actual == 1 and predicted == 1:
            tp_indices.append(plot_idx)
        elif actual == 0 and predicted == 1:
            fp_indices.append(plot_idx)
        elif actual == 1 and predicted == 0:
            fn_indices.append(plot_idx)
            
    if tp_indices:
        plt.scatter(tp_indices, metrics[tp_indices], color="green", label="true positive", zorder=5, marker='o', s=80)
        
    if fp_indices:
        plt.scatter(fp_indices, metrics[fp_indices], color="orange", label="false positive", zorder=6, marker='o', s=80)
        
    if fn_indices:
        plt.scatter(fn_indices, metrics[fn_indices], color="red", label="false negative", zorder=7, marker='o', s=80)

    plt.xlabel("time step")
    plt.ylabel("metric value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
   # plt.xlim(test_start_idx - 50, len(metrics))
    
    plt.tight_layout()
    plt.show()

def generate_data(n_steps=5000, anomaly_prob=0.03, signal_length=10):
    np.random.seed(0)

    # simulating increasing trend and seasonal fluctuations 
    t = np.arange(n_steps)
    trend = 0.002 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 200)
    noise = np.random.normal(0, 3, n_steps)
    
    metrics = 50 + trend + seasonal + noise
    labels = np.zeros(n_steps)
    
    i = signal_length
    while i < n_steps:
        if np.random.rand() < anomaly_prob:
            
            # injecting a pattern before the incident
            for j in range(signal_length):
                metrics[i - signal_length + j] += (j * np.random.uniform(1.0, 3.0)) 
                
            # make about 1/3 of spikes false alarms
            if np.random.rand() < 0.66:
                # if the incident is real, it can differ in length
                duration = np.random.randint(3,10)
                
                # checking if index doesnt go out of the array bounds
                end = min(i + duration, n_steps)
                
                labels[i : end] = 1
                metrics[i : end] += np.random.uniform(30,40)
                i += duration
            else:
                end = min(i + 3, n_steps)
                metrics[i : end] += np.random.uniform(10,20)
            
            # skip a few steps so that incidents dont overlap
            i += signal_length 
        else:
            i += 1
            
    return metrics, labels

def create_sliding_windows(metrics, labels, W, H):
    X, y = [], []
    n_steps = len(metrics)
    
    for i in range(n_steps - W - H + 1):
        window_features = metrics[i : i + W]
        
        mean_val = np.mean(window_features)
        std_val = np.std(window_features)
        min_val = np.min(window_features)
        max_val = np.max(window_features)
        
        delta = window_features[-1] - window_features[0]
        
        enhanced_features = np.concatenate([
            window_features, 
            [mean_val, std_val, min_val, max_val, delta]
        ])
        
        
        future_labels = labels[i + W : i + W + H]
        target = 1 if np.any(future_labels == 1) else 0
        
        X.append(enhanced_features)
        y.append(target)
    
    return np.array(X), np.array(y)

W = 15
H = 3

metrics, labels = generate_data()

X, y = create_sliding_windows(metrics, labels, W, H)

# 80:20 split for train:test
split_index = int(len(X) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

print("raport:")
print(classification_report(y_test, y_pred))

print("confusion matrix:")
print("TN, FP")
print("FN, TP")
print(confusion_matrix(y_test, y_pred))

plot_predictions(metrics, y_test, y_pred, split_index, W)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

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

def create_sliding_windows(metrics, labels, W, H, enhance = 0):
    X, y, features = [], [], []
    n_steps = len(metrics)
    
    if enhance == 1:
        features = ["mean", "std", "min", "max", "delta", "diff_mean", "diff_std", "diff_max", "diff_min", "slope", "distance_from_max", "distance_from_min"]
    elif enhance == 2:
        features = ["std", "delta", "diff_mean", "slope", "distance_from_min"]
    
    for i in range(n_steps - W - H + 1):
        window_features = metrics[i : i + W]
        
        mean_val = np.mean(window_features)
        std_val = np.std(window_features)
        min_val = np.min(window_features)
        max_val = np.max(window_features)
        
        delta = window_features[-1] - window_features[0]
        
        diffs = np.diff(window_features)
        
        diff_mean = np.mean(diffs)
        diff_std = np.std(diffs)
        diff_max = np.max(diffs)
        diff_min = np.min(diffs)
        
        x = np.arange(len(window_features))
        slope = np.polyfit(x, window_features, 1)[0]
        
        distance_from_max = window_features[-1] - np.max(window_features)
        distance_from_min = window_features[-1] - np.min(window_features)
        
        # no enhancements
        if enhance == 0:
            enhanced_features = window_features
        
        # all enhancements
        elif enhance == 1:
            enhanced_features = np.concatenate([
                window_features, 
                [mean_val, std_val, min_val, max_val, delta, diff_mean, diff_std, diff_max, diff_min, slope, distance_from_max, distance_from_min]
            ])
            
        # enhancements with the most importance
        elif enhance == 2:
            enhanced_features = np.concatenate([
                window_features, 
                [std_val, delta, diff_mean, slope, distance_from_min]
            ])
        
        future_labels = labels[i + W : i + W + H]
        target = 1 if np.any(future_labels == 1) else 0
        
        X.append(enhanced_features)
        y.append(target)
    
    return np.array(X), np.array(y), features


metrics, labels = generate_data()

best_f1 = 0
best_params = {}
H = 3

for W in range(5, 31, 5):
    for d in range(0, 3):
        X, y, features = create_sliding_windows(metrics, labels, W, H, d)

        # 60:20:20 split for train:validation:test
        train_idx = int(len(X) * 0.6)
        val_idx = int(len(X) * 0.8)
        
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0)
        clf.fit(X_train_scaled, y_train)

        y_val_probs = clf.predict_proba(X_val_scaled)[:, 1]
        
        for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
            y_val_pred = (y_val_probs >= threshold).astype(int)
            
            current_f1 = f1_score(y_val, y_val_pred, zero_division=0)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = {'W': W, 'H': H, 'd': d, 'threshold': threshold, 'model': clf, 'scaler': scaler}

best_W = best_params['W']
best_H = best_params['H']
best_d = best_params['d']
best_threshold = best_params['threshold']
best_model = best_params['model']
best_scaler = best_params['scaler']



print("optimal parameters:")
print(f"W: {best_W}\n  H: {best_H}\n d: {best_d}\n threshold: {best_threshold}")

X_best, y_best, features = create_sliding_windows(metrics, labels, best_W, best_H, best_d)

val_idx = int(len(X_best) * 0.8)
X_test = X_best[val_idx:]
y_test = y_best[val_idx:]

X_test_scaled = best_scaler.transform(X_test)

y_test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_probs >= best_threshold).astype(int)

print("test set results:")
print(classification_report(y_test, y_test_pred))

print("confusion matrix:")
print("TN, FP")
print("FN, TP")
print(confusion_matrix(y_test, y_test_pred))

feature_names = [f"x{i+1}" for i in range(best_W)]
feature_names += features

importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("features importance:")

for i in range(len(indices)):
    feature_name = feature_names[indices[i]]
    importance = importances[indices[i]]
    print(f"{i+1:2d}. {feature_name:<20}: {importance:.4f}")
    
plot_predictions(metrics, y_test, y_test_pred, val_idx, best_W)
    
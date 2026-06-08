import os
import csv

def compute_adaptive_alpha(fisher_prev, fisher_curr, alpha_min=0.05, alpha_max=0.95):
    """
    Computes the adaptive merge weight based on the traces of the Fisher Information Matrices.
    Safely handles PyTorch tensors and clamps the output.
    """
    # Safely extract scalar sum from tensors
    F_prev = sum(v.sum().item() if hasattr(v, 'sum') else v for v in fisher_prev.values())
    F_curr = sum(v.sum().item() if hasattr(v, 'sum') else v for v in fisher_curr.values())
    
    # Calculate ratio with epsilon to prevent division by zero
    alpha = F_curr / (F_curr + F_prev + 1e-12)
    
    # Clamp to prevent catastrophic overwriting
    alpha = max(alpha_min, min(alpha, alpha_max))
    
    return alpha, F_prev, F_curr

def log_adaptive_alpha(filepath, task_id, F_prev, F_curr, alpha, accuracy=None):
    """
    Appends the adaptive alpha metrics to a CSV file for paper visualizations.
    """
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['task_id', 'F_prev', 'F_curr', 'alpha', 'accuracy'])
        writer.writerow([task_id, f"{F_prev:.4f}", f"{F_curr:.4f}", f"{alpha:.4f}", accuracy if accuracy else ""])
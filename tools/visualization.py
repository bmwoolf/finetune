import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from typing import Dict, List
import os
from pathlib import Path

# Create plots directory in example folder
PLOTS_DIR = Path(__file__).parent.parent / "example" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def plot_training_progress(metrics: Dict[str, List[Dict]], save_plot: bool = True) -> None:
    """Plot training and validation metrics over time."""
    train_df = pd.DataFrame(metrics['train'])
    valid_df = pd.DataFrame(metrics['valid'])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(train_df['step'], train_df['loss'], label='Training Loss', color='blue', linewidth=2)
    axes[0].plot(valid_df['step'], valid_df['loss'], label='Validation Loss', color='red', linewidth=2)
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUPRC progression
    if 'auprc' in valid_df.columns:
        axes[1].plot(valid_df['step'], valid_df['auprc'], label='Validation AUPRC', color='green', linewidth=2)
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('AUPRC')
        axes[1].set_title('AUPRC Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'AUPRC data not available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('AUPRC Over Time')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(PLOTS_DIR / 'training_progress.png', dpi=300, bbox_inches='tight')
        print(f"Saved training progress plot to: {PLOTS_DIR / 'training_progress.png'}")
    
    plt.show()

def plot_function_analysis(overview_df: pd.DataFrame, targets: List[str], 
                         valid_true_df: pd.DataFrame, valid_prob_df: pd.DataFrame, 
                         save_plot: bool = True) -> None:
    """Create comprehensive function performance analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top 10 performing functions
    top_functions = overview_df.head(10)
    y_pos = range(len(top_functions))
    bars = axes[0, 0].barh(y_pos, top_functions['auprc'], color='skyblue', alpha=0.7)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels([desc[:40] + '...' if len(desc) > 40 else desc 
                               for desc in top_functions['description']], fontsize=8)
    axes[0, 0].set_xlabel('AUPRC')
    axes[0, 0].set_title('Top 10 Performing Functions')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[0, 0].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # Function frequency vs performance - FIXED
    # Only use functions that exist in both datasets
    common_functions = set(overview_df.index) & set(targets)
    if len(common_functions) > 0:
        function_counts = valid_true_df[list(common_functions)].sum()
        overview_subset = overview_df.loc[list(common_functions)]
        
        scatter = axes[0, 1].scatter(function_counts, overview_subset['auprc'], 
                                    alpha=0.6, c=overview_subset['auprc'], cmap='viridis', s=50)
        axes[0, 1].set_xlabel('Number of Proteins with Function')
        axes[0, 1].set_ylabel('AUPRC')
        axes[0, 1].set_title('Function Frequency vs Performance')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='AUPRC')
    else:
        axes[0, 1].text(0.5, 0.5, 'No common functions found', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Function Frequency vs Performance')
    
    # Precision-Recall curve for best function
    if len(overview_df) > 0:
        best_function = overview_df.index[0]
        if best_function in valid_true_df.columns and best_function in valid_prob_df.columns:
            precision, recall, _ = metrics.precision_recall_curve(
                valid_true_df[best_function], valid_prob_df[best_function]
            )
            axes[1, 0].plot(recall, precision, linewidth=2, color='purple')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title(f'PR Curve: {best_function[:30]}...')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Best function data not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('PR Curve: Best Function')
    
    # AUPRC distribution
    axes[1, 1].hist(overview_df['auprc'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('AUPRC')
    axes[1, 1].set_ylabel('Number of Functions')
    axes[1, 1].set_title('AUPRC Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    mean_auprc = overview_df['auprc'].mean()
    median_auprc = overview_df['auprc'].median()
    axes[1, 1].axvline(mean_auprc, color='red', linestyle='--', label=f'Mean: {mean_auprc:.3f}')
    axes[1, 1].axvline(median_auprc, color='green', linestyle='--', label=f'Median: {median_auprc:.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(PLOTS_DIR / 'function_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved function analysis plot to: {PLOTS_DIR / 'function_analysis.png'}")
    
    plt.show()

def plot_predictions_analysis(valid_true_df: pd.DataFrame, valid_prob_df: pd.DataFrame, 
                            targets: List[str], save_plot: bool = True) -> None:
    """Visualize model predictions and errors."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prediction confidence distribution
    all_probs = valid_prob_df.values.flatten()
    axes[0, 0].hist(all_probs, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_xlabel('Prediction Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Prediction Confidence Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add statistics
    mean_prob = np.mean(all_probs)
    median_prob = np.median(all_probs)
    axes[0, 0].axvline(mean_prob, color='red', linestyle='--', label=f'Mean: {mean_prob:.3f}')
    axes[0, 0].axvline(median_prob, color='green', linestyle='--', label=f'Median: {median_prob:.3f}')
    axes[0, 0].legend()
    
    # True vs Predicted for a sample function
    if len(targets) > 0:
        sample_function = targets[0]
        if sample_function in valid_true_df.columns and sample_function in valid_prob_df.columns:
            true_vals = valid_true_df[sample_function]
            pred_vals = valid_prob_df[sample_function]
            
            axes[0, 1].scatter(true_vals, pred_vals, alpha=0.6, s=20)
            axes[0, 1].set_xlabel('True Values')
            axes[0, 1].set_ylabel('Predicted Probabilities')
            axes[0, 1].set_title(f'True vs Predicted: {sample_function[:30]}...')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add diagonal line for perfect predictions
            axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Prediction')
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, 'Sample function data not available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('True vs Predicted')
    
    # Error analysis - Confusion matrix
    if len(targets) > 0:
        sample_function = targets[0]
        if sample_function in valid_true_df.columns and sample_function in valid_prob_df.columns:
            true_vals = valid_true_df[sample_function]
            pred_vals = valid_prob_df[sample_function]
            predictions = (pred_vals >= 0.5).astype(int)
            true_vals = true_vals.astype(int)
            
            # Confusion matrix
            cm = metrics.confusion_matrix(true_vals, predictions)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], 
                       cmap='Blues', cbar_kws={'label': 'Count'})
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('True')
            axes[1, 0].set_title('Confusion Matrix')
        else:
            axes[1, 0].text(0.5, 0.5, 'Sample function data not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Confusion Matrix')
    
    # ROC curve
    if len(targets) > 0:
        sample_function = targets[0]
        if sample_function in valid_true_df.columns and sample_function in valid_prob_df.columns:
            true_vals = valid_true_df[sample_function]
            pred_vals = valid_prob_df[sample_function]
            
            fpr, tpr, _ = metrics.roc_curve(true_vals, pred_vals)
            roc_auc = metrics.auc(fpr, tpr)
            axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', 
                           linewidth=2, color='blue')
            axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('ROC Curve')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Sample function data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('ROC Curve')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(PLOTS_DIR / 'predictions_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved predictions analysis plot to: {PLOTS_DIR / 'predictions_analysis.png'}")
    
    plt.show()

def plot_metrics_summary(final_results: pd.DataFrame, save_plot: bool = True) -> None:
    """Plot a summary of final metrics for train/valid/test splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract metrics
    metrics_to_plot = ['loss', 'accuracy', 'auprc', 'auroc']
    available_metrics = [m for m in metrics_to_plot if m in final_results.columns]
    
    if len(available_metrics) > 0:
        # Create grouped bar chart
        x = np.arange(len(final_results))
        width = 0.2
        
        for i, metric in enumerate(available_metrics):
            axes[0].bar(x + i * width, final_results[metric], width, 
                       label=metric, alpha=0.8)
        
        axes[0].set_xlabel('Dataset Split')
        axes[0].set_ylabel('Metric Value')
        axes[0].set_title('Final Metrics by Split')
        axes[0].set_xticks(x + width * (len(available_metrics) - 1) / 2)
        axes[0].set_xticklabels(final_results['split'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # AUPRC comparison
    if 'auprc' in final_results.columns:
        bars = axes[1].bar(final_results['split'], final_results['auprc'], 
                          color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[1].set_ylabel('AUPRC')
        axes[1].set_title('AUPRC by Split')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
    
    # AUROC comparison
    if 'auroc' in final_results.columns:
        bars = axes[2].bar(final_results['split'], final_results['auroc'], 
                          color=['lightgreen', 'orange'], alpha=0.7)
        axes[2].set_ylabel('AUROC')
        axes[2].set_title('AUROC by Split')
        axes[2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(PLOTS_DIR / 'metrics_summary.png', dpi=300, bbox_inches='tight')
        print(f"Saved metrics summary plot to: {PLOTS_DIR / 'metrics_summary.png'}")
    
    plt.show() 
"""
Visualization utilities for model comparison and results
Author: Hanoi Temperature Forecasting Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import seaborn as sns


class ModelVisualizer:
    """
    Create visualizations for model comparison and results.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent1': '#F18F01',
            'accent2': '#C73E1D',
            'light': '#87CEEB',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545'
        }
    
    def plot_model_comparison(self, 
                            comparison_df: pd.DataFrame,
                            figsize: Tuple[int, int] = (16, 12)) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create comprehensive model comparison plots.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            figsize: Figure size (width, height)
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Color palette
        colors = [self.colors['light'], self.colors['success'], 
                 self.colors['warning'], self.colors['danger']]
        
        # 1. RMSE Comparison
        axes[0, 0].barh(comparison_df['Model'], comparison_df['Test RMSE'], 
                       color=colors[0], alpha=0.8)
        axes[0, 0].set_xlabel('Test RMSE', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Model Comparison - RMSE (Lower is Better)', 
                           fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # 2. R² Comparison
        axes[0, 1].barh(comparison_df['Model'], comparison_df['Test R²'], 
                       color=colors[1], alpha=0.8)
        axes[0, 1].set_xlabel('Test R²', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Model Comparison - R² (Higher is Better)', 
                           fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Training Time
        axes[1, 0].barh(comparison_df['Model'], comparison_df['Train Time (s)'], 
                       color=colors[2], alpha=0.8)
        axes[1, 0].set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Model Training Time', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. MAPE Comparison
        axes[1, 1].barh(comparison_df['Model'], comparison_df['Test MAPE (%)'], 
                       color=colors[3], alpha=0.8)
        axes[1, 1].set_xlabel('Test MAPE (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Model Comparison - MAPE (Lower is Better)', 
                           fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_predictions_comparison(self, 
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  model_name: str,
                                  num_samples: int = 200,
                                  figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            num_samples: Number of samples to plot
            figsize: Figure size
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Limit samples for visualization
        n_samples = min(num_samples, len(y_true))
        
        # Plot 1: Time series comparison
        x_axis = np.arange(n_samples)
        axes[0].plot(x_axis, y_true[:n_samples], 
                    label='Actual', linewidth=2, alpha=0.8, 
                    color=self.colors['primary'], marker='o', markersize=3)
        axes[0].plot(x_axis, y_pred[:n_samples], 
                    label='Predicted', linewidth=2, alpha=0.8,
                    color=self.colors['secondary'], marker='s', markersize=3)
        axes[0].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Temperature (scaled)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{model_name} - Predictions vs Actual', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.6, s=50, 
                       color=self.colors['accent1'])
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect prediction')
        
        axes[1].set_xlabel('Actual Temperature (scaled)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Predicted Temperature (scaled)', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{model_name} - Scatter Plot', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_all_models_comparison(self, 
                                 results: Dict[str, Dict[str, Any]], 
                                 y_true: np.ndarray,
                                 num_samples: int = 100,
                                 figsize: Tuple[int, int] = (18, 14)) -> Tuple[plt.Figure, np.ndarray]:
        """
        Compare predictions of all models in subplots.
        
        Args:
            results: Dictionary of model results
            y_true: True values
            num_samples: Number of samples to plot
            figsize: Figure size
            
        Returns:
            Tuple of (figure, axes)
        """
        n_models = len(results)
        cols = 2
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Limit samples
        n_samples = min(num_samples, len(y_true))
        x_axis = np.arange(n_samples)
        
        for idx, (name, result) in enumerate(results.items()):
            y_pred = result['y_pred']
            test_rmse = result['test_rmse']
            test_r2 = result['test_r2']
            
            ax = axes[idx]
            ax.plot(x_axis, y_true[:n_samples], 
                   label='Actual', linewidth=2, alpha=0.8,
                   color=self.colors['primary'])
            ax.plot(x_axis, y_pred[:n_samples], 
                   label='Predicted', linewidth=2, alpha=0.8,
                   color=self.colors['secondary'])
            
            ax.set_xlabel('Sample', fontsize=10)
            ax.set_ylabel('Temperature', fontsize=10)
            ax.set_title(f'{name}\nRMSE: {test_rmse:.4f}, R²: {test_r2:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_training_progress(self, 
                             losses: list,
                             title: str = "Training Loss",
                             figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot training loss over epochs.
        
        Args:
            losses: List of loss values
            title: Plot title
            figsize: Figure size
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, label='Training Loss', 
               linewidth=2, color=self.colors['primary'])
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add minimum loss annotation
        min_loss = min(losses)
        min_epoch = losses.index(min_loss) + 1
        ax.annotate(f'Min: {min_loss:.6f}\nEpoch: {min_epoch}',
                   xy=(min_epoch, min_loss),
                   xytext=(min_epoch + len(losses)*0.1, min_loss + (max(losses) - min_loss)*0.1),
                   arrowprops=dict(arrowstyle='->', color=self.colors['accent2']),
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax
    
    def save_plot(self, fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            filepath: Path to save file
            dpi: Resolution
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {filepath}")
    
    def close_all(self) -> None:
        """Close all matplotlib figures."""
        plt.close('all')
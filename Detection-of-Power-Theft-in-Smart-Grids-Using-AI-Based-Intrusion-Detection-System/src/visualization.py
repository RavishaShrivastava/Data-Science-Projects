"""
Visualization Module for Power Theft Detection System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Comprehensive visualization for power theft detection
    """
    
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        
    def plot_confusion_matrix(self, y_true, y_pred, labels=['Normal', 'Theft'], save_name=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_name and self.save_dir:
            plt.savefig(f"{self.save_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curve(self, y_true, y_proba, save_name=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_name and self.save_dir:
            plt.savefig(f"{self.save_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_precision_recall_curve(self, y_true, y_proba, save_name=None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_name and self.save_dir:
            plt.savefig(f"{self.save_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_training_history(self, history, save_name=None):
        """Plot training history for deep learning models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        if 'val_accuracy' in history.history:
            axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train')
        if 'val_loss' in history.history:
            axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        if 'auc' in history.history:
            axes[1, 0].plot(history.history['auc'], label='Train')
            if 'val_auc' in history.history:
                axes[1, 0].plot(history.history['val_auc'], label='Validation')
            axes[1, 0].set_title('Model AUC', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Precision & Recall
        if 'precision' in history.history and 'recall' in history.history:
            axes[1, 1].plot(history.history['precision'], label='Precision')
            axes[1, 1].plot(history.history['recall'], label='Recall')
            axes[1, 1].set_title('Precision & Recall', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        if save_name and self.save_dir:
            plt.savefig(f"{self.save_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_consumption_pattern(self, df, consumption_col='consumption', 
                                 theft_col='is_theft', save_name=None):
        """Plot consumption patterns with theft indicators"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot normal consumption
        normal_data = df[df[theft_col] == 0]
        theft_data = df[df[theft_col] == 1]
        
        ax.plot(normal_data.index, normal_data[consumption_col], 
               color='blue', alpha=0.6, label='Normal', linewidth=1)
        ax.scatter(theft_data.index, theft_data[consumption_col], 
                  color='red', s=50, label='Theft Detected', zorder=5)
        
        ax.set_title('Power Consumption Pattern with Theft Detection', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Consumption (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_name and self.save_dir:
            plt.savefig(f"{self.save_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names, importances, top_n=20, save_name=None):
        """Plot feature importance"""
        # Sort features by importance
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        if save_name and self.save_dir:
            plt.savefig(f"{self.save_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_alert_timeline(self, alerts_df, save_name=None):
        """Plot alert timeline"""
        if 'timestamp' in alerts_df.columns:
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
            alerts_df = alerts_df.sort_values('timestamp')
        
        fig = go.Figure()
        
        # Color mapping for risk levels
        color_map = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow', 'NORMAL': 'green'}
        
        for risk_level in alerts_df['risk_level'].unique():
            data = alerts_df[alerts_df['risk_level'] == risk_level]
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['risk_score'],
                mode='markers',
                name=risk_level,
                marker=dict(
                    size=10,
                    color=color_map.get(risk_level, 'gray'),
                    line=dict(width=1, color='white')
                )
            ))
        
        fig.update_layout(
            title='Alert Timeline by Risk Level',
            xaxis_title='Time',
            yaxis_title='Risk Score',
            hovermode='closest',
            template='plotly_white'
        )
        
        if save_name and self.save_dir:
            fig.write_html(f"{self.save_dir}/{save_name}.html")
        
        return fig
    
    def create_dashboard(self, metrics_dict, save_name=None):
        """Create interactive dashboard with key metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance', 'Risk Distribution', 
                          'Detection Timeline', 'Confusion Matrix'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # Model Performance
        if 'performance' in metrics_dict:
            perf = metrics_dict['performance']
            fig.add_trace(
                go.Bar(x=list(perf.keys()), y=list(perf.values()), 
                      marker_color='steelblue'),
                row=1, col=1
            )
        
        # Risk Distribution
        if 'risk_distribution' in metrics_dict:
            risk_dist = metrics_dict['risk_distribution']
            fig.add_trace(
                go.Pie(labels=list(risk_dist.keys()), values=list(risk_dist.values())),
                row=1, col=2
            )
        
        # Confusion Matrix
        if 'confusion_matrix' in metrics_dict:
            cm = metrics_dict['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues', showscale=False),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Power Theft Detection Dashboard",
            showlegend=True,
            height=800
        )
        
        if save_name and self.save_dir:
            fig.write_html(f"{self.save_dir}/{save_name}.html")
        
        return fig
    
    def plot_model_comparison(self, results_dict, save_name=None):
        """Compare multiple models"""
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results_dict[model].get(metric, 0) for model in models]
            fig.add_trace(go.Bar(name=metric.capitalize(), x=models, y=values))
        
        fig.update_layout(
            title='Model Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            template='plotly_white'
        )
        
        if save_name and self.save_dir:
            fig.write_html(f"{self.save_dir}/{save_name}.html")
        
        return fig


def plot_consumption_heatmap(df, user_col='user_id', time_col='hour', 
                            consumption_col='consumption', save_path=None):
    """
    Create heatmap of consumption patterns by user and time
    """
    pivot_data = df.pivot_table(
        values=consumption_col,
        index=user_col,
        columns=time_col,
        aggfunc='mean'
    )
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Consumption (kWh)'})
    plt.title('Consumption Heatmap by User and Hour', fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('User ID')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    print("Visualization Module for Power Theft Detection")
    print("Available visualizations:")
    print("- Confusion Matrix")
    print("- ROC Curve")
    print("- Precision-Recall Curve")
    print("- Training History")
    print("- Consumption Patterns")
    print("- Feature Importance")
    print("- Alert Timeline")
    print("- Interactive Dashboard")

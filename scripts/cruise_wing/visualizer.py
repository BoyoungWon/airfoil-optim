"""
Optimization Visualizer for Cruise Wing

결과 시각화 모듈:
- Convergence plot
- Design space exploration
- Polar comparison
- Airfoil shape comparison
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class OptimizationVisualizer:
    """
    Cruise Wing 최적화 결과 시각화
    """
    
    def __init__(self, output_dir: Optional[str] = None,
                 style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Parameters
        ----------
        output_dir : str, optional
            Output directory for saving figures
        style : str
            Matplotlib style
        figsize : tuple
            Default figure size
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")
        
        self.output_dir = Path(output_dir) if output_dir else Path("output/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        
        # Try to set style, fall back to default if not available
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                pass  # Use default style
    
    def plot_convergence(self, history: List[Dict],
                         metric: str = 'L/D',
                         title: Optional[str] = None,
                         save: bool = True) -> plt.Figure:
        """
        Plot optimization convergence history
        
        Parameters
        ----------
        history : list
            Optimization history from optimizer
        metric : str
            Metric to plot
        title : str, optional
            Plot title
        save : bool
            Save figure to file
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        iterations = [h['iteration'] for h in history]
        objectives = [h['objective'] for h in history]
        
        # Plot objective value
        ax1 = axes[0]
        ax1.plot(iterations, objectives, 'b-', alpha=0.6, label='Evaluation')
        
        # Running best
        running_best = np.maximum.accumulate(objectives)
        ax1.plot(iterations, running_best, 'r-', linewidth=2, label='Best so far')
        
        ax1.set_ylabel(metric)
        ax1.legend()
        ax1.set_title(title or f'Optimization Convergence - {metric}')
        ax1.grid(True, alpha=0.3)
        
        # Plot parameters
        ax2 = axes[1]
        params_array = np.array([h['params'] for h in history])
        
        labels = ['m (camber)', 'p (position)', 't (thickness)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i in range(params_array.shape[1]):
            ax2.plot(iterations, params_array[:, i], 
                    color=colors[i], label=labels[i], alpha=0.7)
        
        ax2.set_xlabel('Evaluation')
        ax2.set_ylabel('Parameter Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "convergence.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved: {filepath}")
        
        return fig
    
    def plot_airfoil_comparison(self, 
                                 airfoils: Dict[str, np.ndarray],
                                 title: Optional[str] = None,
                                 save: bool = True) -> plt.Figure:
        """
        Compare airfoil shapes
        
        Parameters
        ----------
        airfoils : dict
            Dictionary of {name: coordinates}
        title : str, optional
            Plot title
        save : bool
            Save figure
            
        Returns
        -------
        Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(airfoils)))
        
        for (name, coords), color in zip(airfoils.items(), colors):
            ax.plot(coords[:, 0], coords[:, 1], 
                   color=color, linewidth=2, label=name)
        
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title(title or 'Airfoil Comparison')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable limits
        ax.set_xlim(-0.05, 1.05)
        y_max = max(np.max(np.abs(c[:, 1])) for c in airfoils.values())
        ax.set_ylim(-y_max * 1.5, y_max * 1.5)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "airfoil_comparison.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved: {filepath}")
        
        return fig
    
    def plot_polar_comparison(self,
                               polars: Dict[str, Dict],
                               title: Optional[str] = None,
                               save: bool = True) -> plt.Figure:
        """
        Compare polars (CL vs alpha, CL vs CD, L/D vs alpha)
        
        Parameters
        ----------
        polars : dict
            Dictionary of {name: polar_data}
        title : str, optional
            Plot title
        save : bool
            Save figure
            
        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(polars)))
        
        for (name, polar), color in zip(polars.items(), colors):
            alpha = polar['alpha']
            cl = polar['CL']
            cd = polar['CD']
            ld = polar.get('L/D', cl / cd)
            
            # CL vs alpha
            axes[0, 0].plot(alpha, cl, color=color, linewidth=2, label=name)
            
            # CD vs alpha
            axes[0, 1].plot(alpha, cd * 10000, color=color, linewidth=2)  # In drag counts
            
            # Drag polar (CL vs CD)
            axes[1, 0].plot(cd * 10000, cl, color=color, linewidth=2)
            
            # L/D vs alpha
            axes[1, 1].plot(alpha, ld, color=color, linewidth=2)
        
        # Labels and formatting
        axes[0, 0].set_xlabel('α (°)')
        axes[0, 0].set_ylabel('$C_L$')
        axes[0, 0].set_title('Lift Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('α (°)')
        axes[0, 1].set_ylabel('$C_D$ (counts)')
        axes[0, 1].set_title('Drag Curve')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('$C_D$ (counts)')
        axes[1, 0].set_ylabel('$C_L$')
        axes[1, 0].set_title('Drag Polar')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('α (°)')
        axes[1, 1].set_ylabel('L/D')
        axes[1, 1].set_title('Lift-to-Drag Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title or 'Polar Comparison', fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "polar_comparison.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved: {filepath}")
        
        return fig
    
    def plot_design_space(self,
                          X_train: np.ndarray,
                          y_train: List[Dict],
                          optimal: Optional[np.ndarray] = None,
                          metric: str = 'L/D',
                          save: bool = True) -> plt.Figure:
        """
        Plot design space exploration
        
        Parameters
        ----------
        X_train : np.ndarray
            Training samples
        y_train : list
            Training outputs
        optimal : np.ndarray, optional
            Optimal point
        metric : str
            Metric for coloring
        save : bool
            Save figure
            
        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract metric values for coloring
        if isinstance(y_train[0], dict):
            if metric == 'L/D':
                values = np.array([
                    y.get('L/D', y.get('CL', 0) / max(y.get('CD', 1), 1e-10))
                    for y in y_train
                ])
            else:
                values = np.array([y.get(metric, 0) for y in y_train])
        else:
            values = np.array(y_train)
        
        param_names = ['m (camber)', 'p (position)', 't (thickness)']
        param_pairs = [(0, 1), (0, 2), (1, 2)]
        
        for ax, (i, j) in zip(axes, param_pairs):
            sc = ax.scatter(X_train[:, i], X_train[:, j], 
                           c=values, cmap='viridis', alpha=0.7, s=50)
            
            if optimal is not None:
                ax.scatter(optimal[i], optimal[j], 
                          c='red', marker='*', s=300, 
                          edgecolors='black', linewidths=2,
                          label='Optimal', zorder=5)
            
            ax.set_xlabel(param_names[i])
            ax.set_ylabel(param_names[j])
            ax.grid(True, alpha=0.3)
            
            if optimal is not None:
                ax.legend()
        
        # Colorbar
        cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', 
                           fraction=0.05, pad=0.15)
        cbar.set_label(metric)
        
        fig.suptitle('Design Space Exploration', fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "design_space.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved: {filepath}")
        
        return fig
    
    def plot_surrogate_validation(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   metric: str = 'L/D',
                                   save: bool = True) -> plt.Figure:
        """
        Plot surrogate model validation (predicted vs actual)
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        metric : str
            Metric name
        save : bool
            Save figure
            
        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Parity plot
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.7, s=50)
        
        # Perfect prediction line
        lims = [
            min(min(y_true), min(y_pred)),
            max(max(y_true), max(y_pred))
        ]
        ax1.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
        
        # ±10% bands
        ax1.fill_between(lims, 
                        [l * 0.9 for l in lims], 
                        [l * 1.1 for l in lims],
                        alpha=0.2, color='green', label='±10%')
        
        ax1.set_xlabel(f'True {metric}')
        ax1.set_ylabel(f'Predicted {metric}')
        ax1.set_title('Surrogate Validation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Error histogram
        ax2 = axes[1]
        errors = (y_pred - y_true) / y_true * 100  # Percentage error
        ax2.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='r', linestyle='--', linewidth=2)
        ax2.axvline(np.mean(errors), color='g', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(errors):.2f}%')
        
        ax2.set_xlabel('Prediction Error (%)')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        fig.suptitle(f'{metric} - R²={r2:.4f}, RMSE={rmse:.4f}', fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "surrogate_validation.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved: {filepath}")
        
        return fig
    
    def create_summary_report(self,
                               result,
                               initial_airfoil: Dict,
                               optimal_airfoil: Dict,
                               save: bool = True) -> plt.Figure:
        """
        Create comprehensive summary report
        
        Parameters
        ----------
        result : OptimizationResult
            Optimization result
        initial_airfoil : dict
            Initial airfoil data
        optimal_airfoil : dict
            Optimal airfoil data
        save : bool
            Save figure
            
        Returns
        -------
        Figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Airfoil comparison (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(initial_airfoil['coords'][:, 0], initial_airfoil['coords'][:, 1],
                'b-', linewidth=2, label=f"Initial: {initial_airfoil['name']}")
        ax1.plot(optimal_airfoil['coords'][:, 0], optimal_airfoil['coords'][:, 1],
                'r-', linewidth=2, label=f"Optimal: {optimal_airfoil['name']}")
        ax1.set_xlabel('x/c')
        ax1.set_ylabel('y/c')
        ax1.set_title('Airfoil Shape Comparison')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        
        # 2. Results summary (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        summary_text = f"""
OPTIMIZATION RESULTS
━━━━━━━━━━━━━━━━━━━━
Status: {'✓ Success' if result.success else '✗ Failed'}
Evaluations: {result.n_evaluations}
Iterations: {result.n_iterations}

INITIAL
━━━━━━━
NACA: {initial_airfoil['name']}
L/D: {initial_airfoil.get('L/D', 'N/A'):.2f}
CL: {initial_airfoil.get('CL', 'N/A'):.4f}
CD: {initial_airfoil.get('CD', 'N/A'):.5f}

OPTIMAL
━━━━━━━
NACA: {optimal_airfoil['name']}
L/D: {optimal_airfoil.get('L/D', 'N/A'):.2f}
CL: {optimal_airfoil.get('CL', 'N/A'):.4f}
CD: {optimal_airfoil.get('CD', 'N/A'):.5f}

IMPROVEMENT
━━━━━━━━━━━
ΔL/D: {((optimal_airfoil.get('L/D', 0) / initial_airfoil.get('L/D', 1)) - 1) * 100:+.1f}%
"""
        ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes,
                fontsize=10, fontfamily='monospace',
                verticalalignment='top')
        
        # 3. Convergence (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        objectives = [h['objective'] for h in result.history]
        ax3.plot(objectives, 'b-', alpha=0.6)
        ax3.plot(np.maximum.accumulate(objectives), 'r-', linewidth=2)
        ax3.set_xlabel('Evaluation')
        ax3.set_ylabel('L/D')
        ax3.set_title('Convergence History')
        ax3.grid(True, alpha=0.3)
        
        # 4. Parameters evolution (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        params_array = np.array([h['params'] for h in result.history])
        for i, (name, color) in enumerate(zip(
            ['m', 'p', 't'], ['#1f77b4', '#ff7f0e', '#2ca02c']
        )):
            ax4.plot(params_array[:, i], color=color, label=name, alpha=0.7)
        ax4.set_xlabel('Evaluation')
        ax4.set_ylabel('Parameter Value')
        ax4.set_title('Parameter Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Bar comparison (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        metrics = ['L/D', 'CL', 'CD×1000']
        initial_vals = [
            initial_airfoil.get('L/D', 0),
            initial_airfoil.get('CL', 0),
            initial_airfoil.get('CD', 0) * 1000
        ]
        optimal_vals = [
            optimal_airfoil.get('L/D', 0),
            optimal_airfoil.get('CL', 0),
            optimal_airfoil.get('CD', 0) * 1000
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax5.bar(x - width/2, initial_vals, width, label='Initial', color='blue', alpha=0.7)
        ax5.bar(x + width/2, optimal_vals, width, label='Optimal', color='red', alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics)
        ax5.set_title('Performance Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Polar comparison (bottom, spans all columns) - if polars available
        ax6 = fig.add_subplot(gs[2, :])
        if 'polar' in initial_airfoil and 'polar' in optimal_airfoil:
            ax6.plot(initial_airfoil['polar']['alpha'], 
                    initial_airfoil['polar']['L/D'],
                    'b-', linewidth=2, label=f"Initial: {initial_airfoil['name']}")
            ax6.plot(optimal_airfoil['polar']['alpha'],
                    optimal_airfoil['polar']['L/D'],
                    'r-', linewidth=2, label=f"Optimal: {optimal_airfoil['name']}")
            ax6.set_xlabel('α (°)')
            ax6.set_ylabel('L/D')
            ax6.set_title('L/D vs Angle of Attack')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Polar data not available',
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=14, color='gray')
            ax6.set_title('L/D vs Angle of Attack')
        
        fig.suptitle('Cruise Wing Optimization Summary', fontsize=16, fontweight='bold')
        
        if save:
            filepath = self.output_dir / "optimization_summary.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved: {filepath}")
        
        return fig

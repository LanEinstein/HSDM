
# Import necessary libraries and modules
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime
import random
from Stage_1 import EnhancedMMNetV2, PairedDataset, create_id_mapping

# Set academic style for matplotlib
sns.set_theme(style="whitegrid")

# Customize plot style
plt.rcParams.update({
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 16,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.autolimit_mode': 'round_numbers',
    'axes.labelpad': 10,
    'legend.frameon': False,
    'legend.fontsize': 10,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

class TextureFilter:
    """A collection of texture filters."""
    
    @staticmethod
    def gaussian_blur(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply Gaussian blur to the image."""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def median_blur(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply median blur to the image."""
        return cv2.medianBlur(img.astype(np.float32), kernel_size)
    
    @staticmethod
    def bilateral_filter(img: np.ndarray, d: int = 9, 
                        sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """Apply bilateral filter to the image."""
        return cv2.bilateralFilter(img.astype(np.float32), d, sigma_color, sigma_space)
    
    @staticmethod
    def laplacian(img: np.ndarray) -> np.ndarray:
        """Apply Laplacian filter to the image."""
        return cv2.Laplacian(img, cv2.CV_32F)
    
    @staticmethod
    def sobel(img: np.ndarray, dx: int = 1, dy: int = 1) -> np.ndarray:
        """Apply Sobel filter to the image."""
        return cv2.Sobel(img, cv2.CV_32F, dx, dy)
    
    @staticmethod
    def gabor_filter(img: np.ndarray, ksize: int = 31, sigma: float = 4.0, 
                     theta: float = 0, lambda_: float = 10.0, gamma: float = 0.5) -> np.ndarray:
        """Apply Gabor filter to the image."""
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, 0, ktype=cv2.CV_32F)
        return cv2.filter2D(img, cv2.CV_32F, kernel)

def setup_logging(save_path: str) -> logging.Logger:
    """Set up logging for the application.
    
    Args:
        save_path (str): Directory to save the log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(save_path, f'finetune_{timestamp}.log')
    
    logger = logging.getLogger('FinetuneLogger')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def save_results(results: Dict, save_path: str):
    """Save analysis results to a JSON file.
    
    Args:
        results (Dict): Results to be saved.
        save_path (str): Directory to save the results.
    """
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj
    
    results_processed = convert_for_json(results)
    with open(os.path.join(save_path, 'finetune_results.json'), 'w') as f:
        json.dump(results_processed, f, indent=4)

class FilteredDataset(Dataset):
    """Dataset class with added filter processing.
    
    Args:
        csv_file (str): Path to the CSV file containing dataset information.
        id_mapping (dict): Mapping of IDs to labels.
        filter_fn (Optional[Callable]): Function to apply filter to the images.
        brightness_config (Optional[Dict]): Configuration for brightness adjustment.
    """
    def __init__(self, csv_file: str, id_mapping: dict, filter_fn=None, brightness_config=None):
        self.filter_fn = filter_fn
        self.texture_filter = TextureFilter()
        self.dataset = PairedDataset(
            csv_file=csv_file,
            color_mode='Green',
            brightness_config=brightness_config,
            id_mapping=id_mapping
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original data
        onset, apex, bdi_score, id_label = self.dataset[idx]
        
        # Apply filter if specified
        if self.filter_fn is not None:
            # Convert tensor to numpy array and apply filter
            onset_np = onset.squeeze().numpy()
            apex_np = apex.squeeze().numpy()
            
            onset_filtered = self.filter_fn(onset_np)
            apex_filtered = self.filter_fn(apex_np)
            
            # Ensure values are in the range [0, 1]
            onset_filtered = (onset_filtered - onset_filtered.min()) / (onset_filtered.max() - onset_filtered.min() + 1e-8)
            apex_filtered = (apex_filtered - apex_filtered.min()) / (apex_filtered.max() - apex_filtered.min() + 1e-8)
            
            # Convert back to tensor
            onset = torch.from_numpy(onset_filtered).unsqueeze(0).float()
            apex = torch.from_numpy(apex_filtered).unsqueeze(0).float()
        
        return onset, apex, bdi_score, id_label
    
    def get_max_label(self):
        return self.dataset.get_max_label()

class FilterTrainer:
    """Trainer for fine-tuning models with different filters.
    
    Args:
        base_model_path (str): Path to the pre-trained model.
        device (torch.device): Device to run the training on.
    """
    def __init__(self, base_model_path: str, device: torch.device):
        self.device = device
        self.filters = {
            'gaussian_blur': lambda x: TextureFilter.gaussian_blur(x, 3),
            'median_blur': lambda x: TextureFilter.median_blur(x, 3),
            'bilateral': lambda x: TextureFilter.bilateral_filter(x),
            'laplacian': TextureFilter.laplacian,
            'sobel_x': lambda x: TextureFilter.sobel(x, 1, 0),
            'sobel_y': lambda x: TextureFilter.sobel(x, 0, 1),
            'gabor_0': lambda x: TextureFilter.gabor_filter(x, theta=0),
            'gabor_45': lambda x: TextureFilter.gabor_filter(x, theta=np.pi/4),
            'gabor_90': lambda x: TextureFilter.gabor_filter(x, theta=np.pi/2),
            'gabor_135': lambda x: TextureFilter.gabor_filter(x, theta=3*np.pi/4)
        }
        self.base_model_path = base_model_path
        self.id_mapping = None
        self.max_label = None
        self.setup_data()
    
    def setup_data(self):
        """Set up data loaders."""
        # Set CSV paths
        train_csv = 'BDI-II_train2013.csv'
        test_csv = 'BDI-II_2013test.csv'
        
        # Create ID mapping
        id_mapping = create_id_mapping(train_csv, test_csv)
        self.id_mapping = id_mapping
        
        # Create base datasets (without filters)
        self.base_train_dataset = FilteredDataset(train_csv, id_mapping)
        self.base_test_dataset = FilteredDataset(test_csv, id_mapping)
        
        # Record maximum label value
        self.max_label = max(
            self.base_train_dataset.get_max_label(),
            self.base_test_dataset.get_max_label()
        )
    
    def create_model(self) -> nn.Module:
        """Create and load the pre-trained model.
        
        Returns:
            nn.Module: The loaded model.
        """
        # Load checkpoint
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        
        # Create model
        model = EnhancedMMNetV2(
            num_ids=len(self.id_mapping),
            max_label=self.max_label,
            input_channels=1,
            device=self.device
        ).to(self.device)
        
        # Load pre-trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def create_dataloaders(self, filter_fn) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for a specific filter.
        
        Args:
            filter_fn (Callable): Filter function to apply to the images.
        
        Returns:
            Tuple[DataLoader, DataLoader]: Train and test data loaders.
        """
        brightness_config = {
        'target_brightness': 0.5,
        'method': 'lab',
        'strength': 0.8
    }
        # Create datasets with filters
        train_dataset = FilteredDataset(
            'BDI-II_train2013.csv',
            self.id_mapping,
            filter_fn,
            brightness_config=brightness_config
        )
        
        test_dataset = FilteredDataset(
            'BDI-II_2013test.csv',
            self.id_mapping,
            filter_fn,
            brightness_config=brightness_config
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                  optimizer: torch.optim.Optimizer, criterion_huber: nn.Module,
                  criterion_mi: nn.Module, config: dict, epoch: int,
                  logger: logging.Logger) -> Dict:
        """Train the model for one epoch.
        
        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): Data loader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion_huber (nn.Module): Huber loss function.
            criterion_mi (nn.Module): Cross-entropy loss function.
            config (dict): Configuration dictionary.
            epoch (int): Current epoch number.
            logger (logging.Logger): Logger for logging information.
        
        Returns:
            Dict: Training metrics.
        """
        model.train()
        predictions = []
        targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for onset, apex, bdi_score, id_label in pbar:
            onset = onset.float().to(self.device)
            apex = apex.float().to(self.device)
            bdi_score = bdi_score.float().to(self.device)
            id_label = id_label.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, bdi_pred, id_pred, _ = model(onset, apex, id_label)
            
            # Compute loss
            bdi_loss = criterion_huber(bdi_pred.squeeze(), bdi_score)
            mi_loss = criterion_mi(id_pred, id_label)
            total_loss = bdi_loss + config['beta_mi'] * mi_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record predictions
            predictions.extend(bdi_pred.squeeze().detach().cpu().numpy())
            targets.extend(bdi_score.cpu().numpy())
            
            # Update progress bar
            lr = optimizer.param_groups[0]['lr']
            mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'BDI': f"{bdi_loss.item():.4f}",
                'MI': f"{mi_loss.item():.4f}",
                'MAE': f"{mae:.4f}",
                'LR': f"{lr:.2e}"
            })
        
        # Compute metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean(np.square(predictions - targets)))
        
        return {'mae': mae, 'rmse': rmse}
    
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict:
        """Evaluate the model performance.
        
        Args:
            model (nn.Module): The model to evaluate.
            loader (DataLoader): Data loader for evaluation data.
        
        Returns:
            Dict: Evaluation metrics.
        """
        model.eval()
        predictions = []
        targets = []
        attentions = []
        
        with torch.no_grad():
            for onset, apex, bdi_score, _ in loader:
                onset = onset.float().to(self.device)
                apex = apex.float().to(self.device)
                bdi_score = bdi_score.float().to(self.device)
                
                _, bdi_pred, _, attention = model(onset, apex, None)
                predictions.extend(bdi_pred.squeeze().cpu().numpy())
                targets.extend(bdi_score.cpu().numpy())
                attentions.extend(attention.cpu().numpy())
        
        # Compute metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        attentions = np.array(attentions)
        
        errors = np.abs(predictions - targets)
        stats = {
            'predictions': predictions,
            'targets': targets,
            'errors': errors,
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean(np.square(predictions - targets))),
            'std': np.std(errors),
            'attention_mean': np.mean(attentions, axis=0)
        }
        
        return stats
    
    def finetune_filter(self, filter_name: str, save_dir: str,
                       logger: logging.Logger) -> Dict:
        """Fine-tune the model with a specific filter.
        
        Args:
            filter_name (str): Name of the filter to use.
            save_dir (str): Directory to save the results.
            logger (logging.Logger): Logger for logging information.
        
        Returns:
            Dict: Best metrics after fine-tuning.
        """
        # Create save directory
        filter_save_dir = os.path.join(save_dir, filter_name)
        os.makedirs(filter_save_dir, exist_ok=True)
        
        # Configuration parameters
        config = {
            'num_epochs': 200,
            'learning_rate': 4e-4,
            'beta_mi': 0.5,
            'huber_delta': 2.0,
            'brightness_configs': [
            {'target_brightness': 0.5, 'method': 'lab', 'strength': 0.8}
        ]
        }
        
        # Create data loaders
        train_loader, test_loader = self.create_dataloaders(self.filters[filter_name])
        
        # Create model and optimizer
        model = self.create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-7
        )
        
        # Set loss functions
        criterion_huber = nn.SmoothL1Loss(beta=config['huber_delta'])
        criterion_mi = nn.CrossEntropyLoss()
        
        # Record best performance
        best_metrics = {
            'epoch': 0,
            'mae': float('inf'),
            'model_state': None
        }
        
        # Training loop
        for epoch in range(config['num_epochs']):
            # Train
            train_metrics = self.train_epoch(
                model, train_loader, optimizer,
                criterion_huber, criterion_mi,
                config, epoch, logger
            )
            
            # Evaluate
            test_metrics = self.evaluate(model, test_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log information
            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            logger.info(f"Training MAE: {train_metrics['mae']:.4f}")
            logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
            
            # Update best model
            if test_metrics['mae'] < best_metrics['mae']:
                best_metrics.update({
                    'epoch': epoch,
                    'mae': test_metrics['mae'],
                    'rmse': test_metrics['rmse'],
                    'model_state': model.state_dict().copy(),
                    'attention_mean': test_metrics['attention_mean']
                })
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mae': test_metrics['mae'],
                    'config': config
                }, os.path.join(filter_save_dir, 'best_model.pth'))
                
                logger.info(f"New best model saved! MAE: {best_metrics['mae']:.4f}")
            
            # Update learning rate
            scheduler.step()
        
        return best_metrics

def plot_results(results: Dict, save_path: str):
    """Plot analysis results.
    
    Args:
        results (Dict): Results to plot.
        save_path (str): Directory to save the plots.
    """
    def plot_error_distribution(base_performance: Dict, filter_results: Dict, save_path: str):
        """Plot error distribution.
        
        Args:
            base_performance (Dict): Base model performance.
            filter_results (Dict): Filtered model performance.
            save_path (str): Directory to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original error distribution
        base_errors = base_performance['errors']
        sns.histplot(
            data=base_errors,
            ax=ax1,
            color='lightblue',
            bins=30,
            kde=True,
            stat='count'
        )
        ax1.set_title('Original Error Distribution')
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Mean errors for each filter (sorted by MAE)
        filter_names = []
        mean_errors = []
        error_stds = []
        
        for filter_name, stats in sorted(filter_results.items(), key=lambda x: x[1]['mae']):
            filter_names.append(filter_name)
            mean_errors.append(stats['mae'])
            error_stds.append(stats['std'])
        
        # Error bar plot (sorted by MAE)
        ax2.errorbar(
            mean_errors,
            filter_names,
            xerr=error_stds,
            fmt='o',
            capsize=5,
            color='darkblue',
            markersize=5
        )
        ax2.set_title('Mean Error by Filter (with std)')
        ax2.set_xlabel('Mean Absolute Error')
        ax2.grid(True, alpha=0.3)
        
        # Add vertical line for original MAE
        ax2.axvline(
            x=base_performance['mae'],
            color='r',
            linestyle='--',
            alpha=0.5,
            label='Original MAE'
        )
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'error_distribution.png'),
                   bbox_inches='tight', dpi=330)
        plt.close()
    
    def plot_attention_patterns(base_performance: Dict, filter_results: Dict, save_path: str):
        """Plot attention patterns.
        
        Args:
            base_performance (Dict): Base model performance.
            filter_results (Dict): Filtered model performance.
            save_path (str): Directory to save the plot.
        """
        n_filters = len(filter_results) + 1
        n_cols = 4
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(16, 4 * n_rows))
        gs = plt.GridSpec(n_rows, n_cols, figure=fig)
        gs.update(wspace=0.3, hspace=0.4)
        
        def reshape_attention(attention):
            if len(attention.shape) == 1:
                attention_1d = attention.reshape(1, -1)
                attention_2d = cv2.resize(
                    attention_1d,
                    (16, 8),
                    interpolation=cv2.INTER_LINEAR
                )
                return attention_2d
            return attention
        
        # Get value range for all attention maps
        all_attentions = [base_performance['attention_mean']]
        all_attentions.extend([stats['attention_mean'] for stats in filter_results.values()])
        vmin = min(att.min() for att in all_attentions)
        vmax = max(att.max() for att in all_attentions)
        
        # Plot original attention map
        ax = fig.add_subplot(gs[0, 0])
        base_attention = reshape_attention(base_performance['attention_mean'])
        im = ax.imshow(base_attention, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title('Original', pad=10)
        ax.axis('off')
        
        # Plot attention maps for each filter
        for i, (filter_name, stats) in enumerate(sorted(
            filter_results.items(), key=lambda x: x[1]['mae'])):  # Sort by MAE
            row = (i + 1) // n_cols
            col = (i + 1) % n_cols
            ax = fig.add_subplot(gs[row, col])
            attention = reshape_attention(stats['attention_mean'])
            ax.imshow(attention, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f"{filter_name}\nMAE: {stats['mae']:.4f}", pad=10)  # Add MAE info
            ax.axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.savefig(os.path.join(save_path, 'attention_patterns.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    # Execute plotting
    plot_error_distribution(results['base_performance'], results['filter_results'], save_path)
    plot_attention_patterns(results['base_performance'], results['filter_results'], save_path)

def main():
    """Main function: Execute the full fine-tuning and analysis pipeline."""
    # Basic setup
    save_dir = './FilterFinetuneResults2013'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Set up logging
    logger = setup_logging(save_dir)
    logger.info("Starting filter fine-tuning process")
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize trainer
        base_model_path = './Stage2_Final/stage_two_2013_Green_bright_0.5_huber_2.0_20241204_164643/best_model.pth'
        trainer = FilterTrainer(base_model_path, device)
        logger.info("Initialized trainer")
        
        # Get base performance
        logger.info("Evaluating base performance...")
        base_model = trainer.create_model()
        _, test_loader = trainer.create_dataloaders(None)
        base_performance = trainer.evaluate(base_model, test_loader)
        logger.info(f"Base MAE: {base_performance['mae']:.4f}")
        
        # Fine-tune each filter
        filter_results = {}
        for filter_name in trainer.filters.keys():
            logger.info(f"\nFine-tuning {filter_name} filter...")
            try:
                best_metrics = trainer.finetune_filter(filter_name, save_dir, logger)
                filter_results[filter_name] = {
                    'mae': best_metrics['mae'],
                    'rmse': best_metrics['rmse'],
                    'std': best_metrics.get('std', 0),
                    'attention_mean': best_metrics['attention_mean']
                }
                logger.info(f"Completed fine-tuning {filter_name}")
                logger.info(f"Best MAE: {best_metrics['mae']:.4f}")
            except Exception as e:
                logger.error(f"Error fine-tuning {filter_name}: {str(e)}")
                continue
        
        # Organize results
        results = {
            'base_performance': base_performance,
            'filter_results': filter_results
        }
        
        # Save results
        save_results(results, save_dir)
        
        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        plot_results(results, save_dir)
        
        logger.info("\nProcess completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during process: {str(e)}")
        raise

if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()
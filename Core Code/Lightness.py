import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from Stage_1 import (
    PairedDataset,
    EnhancedMMNetV2,
    create_id_mapping
)

class GradientAnalyzer:
    """
    Brightness gradient analyzer for analyzing depression-related features.
    """

    def __init__(self, 
                 model_path: str,
                 save_dir: str = './Light_Grad',
                 device: torch.device = torch.device('cuda:0')):
        """
        Initialize the gradient analyzer.

        Args:
            model_path (str): Path to the model checkpoint.
            save_dir (str): Directory to save analysis results. Default is './Light_Grad'.
            device (torch.device): Device to run computations on. Default is 'cuda:0'.
        """
        self.device = device
        self.save_dir = save_dir
        self.gradient_dir = os.path.join(save_dir, 'gradient_analysis')
        os.makedirs(self.gradient_dir, exist_ok=True)

        self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load the depression detection model and setup related configurations.

        Args:
            model_path (str): Path to the model checkpoint.
        """
        print(f"Loading model from: {model_path}")
        self.checkpoint = torch.load(model_path, map_location=self.device)

        train_csv = 'BDI-II_train2014.csv'
        test_csv = 'BDI-II_2014test.csv'
        self.id_mapping = create_id_mapping(train_csv, test_csv)

        test_dataset = PairedDataset(
            csv_file=test_csv,
            color_mode='Red',
            brightness_config=None,
            id_mapping=self.id_mapping
        )

        num_ids = len(self.id_mapping)
        max_label = test_dataset.get_max_label()

        self.model = EnhancedMMNetV2(
            num_ids=num_ids,
            max_label=max_label,
            input_channels=1,
            device=self.device
        ).to(self.device)

        if 'model_state_dict' in self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(self.checkpoint)
        self.model.eval()

    def set_paper_style(self):
        """
        Configure visualization style for plots.
        """
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'figure.figsize': (12, 5),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'axes.linewidth': 0.5,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'legend.frameon': False,
            'legend.borderpad': 0.4,
            'axes.prop_cycle': plt.cycler('color', 
                ['#4292C6', '#E41A1C', '#4DAF4A', '#984EA3', '#FF7F00']),
            'figure.autolayout': True,
            'axes.axisbelow': True,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

    def compute_basic_brightness_gradient(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the basic brightness gradient for an image.

        Args:
            image (np.ndarray): Input image as a 2D array.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Gradient magnitude (2D array).
                - Gradient direction (2D array).
        """
        if image.max() > 1.0:
            image = image / 255.0

        smoothed = cv2.GaussianBlur(image, (5, 5), 0)
        grad_x = cv2.Scharr(smoothed, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(smoothed, cv2.CV_64F, 0, 1)

        grad_x = cv2.GaussianBlur(grad_x, (3, 3), 0)
        grad_y = cv2.GaussianBlur(grad_y, (3, 3), 0)

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)

        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        return gradient_magnitude, gradient_direction

    def analyze_and_visualize(self, dataloader: DataLoader):
        """
        Perform analysis and generate visualizations.

        Args:
            dataloader (DataLoader): DataLoader providing the dataset for analysis.
        """
        print("Analyzing brightness distribution...")
        brightness_stats = self.analyze_brightness_distribution(dataloader)

        print("Analyzing gradients...")
        gradient_stats = self.analyze_gradients(dataloader)

        print("Generating visualizations...")
        self.plot_brightness_distribution(brightness_stats)
        self.plot_brightness_severity_correlation(brightness_stats)
        self.plot_gradient_severity_correlation(gradient_stats)
        self.plot_gradient_direction_distribution(gradient_stats)

        print("Saving analysis results...")
        self.save_analysis_report(brightness_stats, gradient_stats)

        np.savez(os.path.join(self.gradient_dir, 'analysis_data.npz'),
                 brightness_stats=brightness_stats,
                 gradient_stats=gradient_stats)

        print(f"Analysis complete! Results saved in: {self.gradient_dir}")

def main():
    """
    Main execution function for running the gradient analysis pipeline.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    params = {
        'model_path': './Stage2_Final/stage_two_2014_Red_original_huber_13.0_20241128_131000/best_model.pth',
        'save_dir': './Light_Grad',
        'device': device
    }

    try:
        analyzer = GradientAnalyzer(**params)

        test_dataset = PairedDataset(
            csv_file='BDI-II_2014test.csv',
            color_mode='Red',
            brightness_config=None,
            id_mapping=analyzer.id_mapping
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=12,
            pin_memory=True
        )

        analyzer.analyze_and_visualize(test_loader)

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main()
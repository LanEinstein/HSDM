import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import json
from datetime import datetime
from torch.utils.data import DataLoader
from Stage_1 import *

class RegionMaskGenerator:
    """
    Generates facial region masks based on different segmentation methods.
    """

    @staticmethod
    def get_three_part_masks(size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Generate masks dividing the image into three parts (upper, middle, lower).

        Args:
            size (Tuple[int, int]): Image size as (height, width).

        Returns:
            Dict[str, np.ndarray]: Dictionary of masks for each region.
        """
        H, W = size
        h_unit = H // 3

        masks = {
            'upper': np.zeros((H, W), dtype=bool),
            'middle': np.zeros((H, W), dtype=bool),
            'lower': np.zeros((H, W), dtype=bool)
        }

        masks['upper'][0:h_unit, :] = True
        masks['middle'][h_unit:2 * h_unit, :] = True
        masks['lower'][2 * h_unit:, :] = True

        return masks

    @staticmethod
    def get_five_part_masks(size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Generate masks dividing the image into five parts (forehead, eyes, nose, mouth, chin).

        Args:
            size (Tuple[int, int]): Image size as (height, width).

        Returns:
            Dict[str, np.ndarray]: Dictionary of masks for each region.
        """
        H, W = size
        h_unit = H // 5

        masks = {
            'forehead': np.zeros((H, W), dtype=bool),
            'eyes': np.zeros((H, W), dtype=bool),
            'nose': np.zeros((H, W), dtype=bool),
            'mouth': np.zeros((H, W), dtype=bool),
            'chin': np.zeros((H, W), dtype=bool)
        }

        masks['forehead'][0:h_unit, :] = True
        masks['eyes'][h_unit:2 * h_unit, :] = True
        masks['nose'][2 * h_unit:3 * h_unit, :] = True
        masks['mouth'][3 * h_unit:4 * h_unit, :] = True
        masks['chin'][4 * h_unit:, :] = True

        return masks

class MaskApplier:
    """
    Applies different masking methods to the input image.
    """

    @staticmethod
    def apply_zero_mask(image: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """
        Remove region information by setting pixel values to zero.

        Args:
            image (torch.Tensor): Input image tensor (C, H, W).
            mask (np.ndarray): Boolean mask indicating the region to mask.

        Returns:
            torch.Tensor: Masked image.
        """
        masked = image.clone()
        if len(masked.shape) == 3:
            for c in range(masked.shape[0]):
                masked[c][mask] = 0
        else:
            masked[mask] = 0
        return masked

    @staticmethod
    def apply_blur_mask(image: torch.Tensor, mask: np.ndarray, kernel_size: int = 11) -> torch.Tensor:
        """
        Retain low-frequency information by applying a Gaussian blur to the region.

        Args:
            image (torch.Tensor): Input image tensor (C, H, W).
            mask (np.ndarray): Boolean mask indicating the region to blur.
            kernel_size (int): Size of the Gaussian kernel. Default is 11.

        Returns:
            torch.Tensor: Masked image.
        """
        masked = image.clone()
        if len(masked.shape) == 3:
            for c in range(masked.shape[0]):
                channel = masked[c].cpu().numpy()
                blurred = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
                channel[mask] = blurred[mask]
                masked[c] = torch.from_numpy(channel)
        else:
            img_np = masked.cpu().numpy()
            blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
            img_np[mask] = blurred[mask]
            masked = torch.from_numpy(img_np)
        return masked

    @staticmethod
    def apply_mean_mask(image: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """
        Retain average brightness for the masked region.

        Args:
            image (torch.Tensor): Input image tensor (C, H, W).
            mask (np.ndarray): Boolean mask indicating the region.

        Returns:
            torch.Tensor: Masked image.
        """
        masked = image.clone()
        if len(masked.shape) == 3:
            for c in range(masked.shape[0]):
                mean_value = masked[c][mask].mean()
                masked[c][mask] = mean_value
        else:
            mean_value = masked[mask].mean()
            masked[mask] = mean_value
        return masked

class RegionAnalyzer:
    """
    Analyzes facial regions using masks and computes their importance for predictions.
    """
    def __init__(self, 
                 model_path: str,
                 device: torch.device = torch.device('cuda:0'),
                 save_dir: str = './region_analysis_results'):
        """
        Initialize the RegionAnalyzer.

        Args:
            model_path (str): Path to the model checkpoint.
            device (torch.device): Device to run the model. Default is CUDA device 0.
            save_dir (str): Directory to save analysis results. Default is './region_analysis_results'.
        """
        self.device = device
        self.save_dir = save_dir
        self.mask_generator = RegionMaskGenerator()
        self.mask_applier = MaskApplier()

        os.makedirs(save_dir, exist_ok=True)

        print(f"Loading checkpoint from: {model_path}")
        self.checkpoint = torch.load(model_path, map_location=device)
        self.config = self.checkpoint.get('config', {})
        self.brightness_config = self.checkpoint.get('brightness_config')

        print("Setting up model...")
        self.setup_model()

    def setup_model(self):
        """
        Set up the model for analysis.
        """
        train_csv = 'BDI-II_train2014.csv'
        test_csv = 'BDI-II_2014test.csv'

        self.id_mapping = create_id_mapping(train_csv, test_csv)

        test_dataset = PairedDataset(
            csv_file=test_csv,
            color_mode='Red',
            brightness_config=self.brightness_config,
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

        try:
            if 'model_state_dict' in self.checkpoint:
                print("Loading state dict from checkpoint...")
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
            else:
                print("Loading raw checkpoint...")
                self.model.load_state_dict(self.checkpoint)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        self.model.eval()

    def analyze_sample(self, 
                       onset: torch.Tensor, 
                       apex: torch.Tensor,
                       mask_methods: List[str] = ['zero', 'blur', 'mean'],
                       mask_types: List[str] = ['three_part', 'five_part']) -> Dict:
        """
        Analyze the importance of different facial regions for a given sample.

        Args:
            onset (torch.Tensor): Onset image tensor.
            apex (torch.Tensor): Apex image tensor.
            mask_methods (List[str]): Masking methods to apply. Default includes 'zero', 'blur', and 'mean'.
            mask_types (List[str]): Types of masks to apply. Default includes 'three_part' and 'five_part'.

        Returns:
            Dict: Analysis results with performance changes.
        """
        results = {
            'static': {'onset': {}, 'apex': {}},
            'dynamic': {}
        }

        with torch.no_grad():
            original_pred = self.model(
                onset.unsqueeze(0).to(self.device), 
                apex.unsqueeze(0).to(self.device), 
                None
            )[1].squeeze().cpu().item()

        for image_type, image in [('onset', onset), ('apex', apex)]:
            for mask_type in mask_types:
                results['static'][image_type][mask_type] = {}

                if mask_type == 'three_part':
                    masks = self.mask_generator.get_three_part_masks(image.shape[-2:])
                else:
                    masks = self.mask_generator.get_five_part_masks(image.shape[-2:])

                for mask_name, mask in masks.items():
                    results['static'][image_type][mask_type][mask_name] = {}

                    for method in mask_methods:
                        if method == 'zero':
                            masked_image = self.mask_applier.apply_zero_mask(image, mask)
                        elif method == 'blur':
                            masked_image = self.mask_applier.apply_blur_mask(image, mask)
                        else:
                            masked_image = self.mask_applier.apply_mean_mask(image, mask)

                        if image_type == 'onset':
                            pred_input = (masked_image.unsqueeze(0).to(self.device), 
                                          apex.unsqueeze(0).to(self.device))
                        else:
                            pred_input = (onset.unsqueeze(0).to(self.device), 
                                          masked_image.unsqueeze(0).to(self.device))

                        with torch.no_grad():
                            masked_pred = self.model(*pred_input, None)[1].squeeze().cpu().item()

                        perf_change = abs(masked_pred - original_pred)
                        results['static'][image_type][mask_type][mask_name][method] = perf_change

        return results

def main():
    """
    Main function to perform region-based analysis on a dataset.
    """
    model_path = './path/to/model_checkpoint.pth'
    device = torch.device('cuda:0')
    save_dir = './analysis_results'

    test_dataset = PairedDataset(
        csv_file='test.csv',
        color_mode='Red',
        brightness_config=None,
        id_mapping=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    analyzer = RegionAnalyzer(
        model_path=model_path,
        device=device,
        save_dir=save_dir
    )

    analyzer.analyze_dataset(test_loader)

if __name__ == '__main__':
    main()

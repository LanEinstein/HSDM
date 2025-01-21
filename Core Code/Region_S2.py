import os
import torch
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from Stage_1 import (
    PairedDataset,
    EnhancedMMNetV2,
    create_id_mapping
)

class DepressionAnalyzer:
    def __init__(self, 
                 model_path: str,
                 save_dir: str = './depression_analysis_results',
                 device: torch.device = torch.device('cuda:1')):
        """
        Unified analyzer for depression patterns with optimized window settings
        """
        self.device = device
        self.save_dir = save_dir
        self.maps_dir = os.path.join(save_dir, 'contribution_maps')
        self.viz_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(self.maps_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Optimized analysis parameters - using consistent 8x8 window
        self.window_size = 8  # 8x8 window for both static and dynamic
        self.stride = 1       # Optimal stride of half window size
        
        # Initialize face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Face contour indices
        self.FACE_CONTOUR_INDICES = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        # Colors for severity levels
        self.severity_colors = {
            'None (0-13)': np.array([33/255, 102/255, 172/255]),
            'Mild (14-19)': np.array([146/255, 197/255, 222/255]),
            'Moderate (20-28)': np.array([244/255, 165/255, 130/255]),
            'Severe (29-63)': np.array([178/255, 24/255, 43/255])
        }
        
        self.load_model(model_path)
        self.template_landmarks = None
        self.template_image = None
        self.template_mask = None

    def load_model(self, model_path: str):
        """Load depression detection model"""
        print(f"Loading model from: {model_path}")
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.config = self.checkpoint.get('config', {})
        self.brightness_config = self.checkpoint.get('brightness_config')
        
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
        
        if 'model_state_dict' in self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(self.checkpoint)
        self.model.eval()

    def get_facial_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks with proper preprocessing"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Convert to uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        
        h, w = image_rgb.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h] 
            for lm in results.multi_face_landmarks[0].landmark
        ])
        
        return landmarks

    def create_face_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create binary mask for face region"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        contour_points = landmarks[self.FACE_CONTOUR_INDICES].astype(np.int32)
        cv2.fillPoly(mask, [contour_points], 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def enhance_template(self):
        """Enhance template image quality"""
        template = self.template_image.copy()
        template = np.clip(template * 1.2, 0, 1)
        
        temp_uint8 = (template * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(temp_uint8)
        enhanced = cv2.fastNlMeansDenoising(enhanced)
        
        self.template_image = enhanced.astype(np.float32) / 255

    def select_template(self, dataloader: DataLoader) -> None:
        """Select best template face"""
        best_score = -1
        samples_checked = 0
        valid_detections = 0
        
        print("\nSelecting optimal template face...")
        for onset, apex, _, _ in tqdm(dataloader, desc="Evaluating faces"):
            samples_checked += 1
            image = apex.squeeze().numpy()
            landmarks = self.get_facial_landmarks(image)
            
            if landmarks is None:
                continue
                
            valid_detections += 1
            try:
                face_mask = self.create_face_mask(landmarks, image.shape)
                face_area = np.sum(face_mask)
                total_area = image.shape[0] * image.shape[1]
                visibility_ratio = face_area / total_area
                
                if visibility_ratio < 0.15:  # Face too small
                    continue
                
                center_y = np.mean(landmarks[:, 1])
                if center_y > image.shape[0] * 0.7:  # Face too low
                    continue
                
                symmetry_score = self.calculate_symmetry_score(landmarks)
                coverage_score = self.calculate_face_coverage(landmarks, image.shape)
                
                # Combined score
                score = symmetry_score * coverage_score * visibility_ratio
                
                if score > best_score:
                    best_score = score
                    self.template_image = image
                    self.template_landmarks = landmarks
                    self.template_mask = face_mask
                    print(f"\nFound better template (sample {samples_checked}):")
                    print(f"Score: {score:.4f}, Symmetry: {symmetry_score:.4f}")
                    print(f"Coverage: {coverage_score:.4f}, Visibility: {visibility_ratio:.4f}")
                    
            except Exception as e:
                continue
        
        if self.template_image is None:
            raise ValueError("Failed to find suitable template face")
        
        self.enhance_template()
        self.save_template_visualization()

    def calculate_symmetry_score(self, landmarks: np.ndarray) -> float:
        """Calculate face symmetry score"""
        center_x = np.mean(landmarks[:, 0])
        left_indices = self.FACE_CONTOUR_INDICES[:len(self.FACE_CONTOUR_INDICES)//2]
        right_indices = self.FACE_CONTOUR_INDICES[len(self.FACE_CONTOUR_INDICES)//2:]
        
        left_points = landmarks[left_indices]
        right_points = landmarks[right_indices]
        
        right_points_mirrored = right_points.copy()
        right_points_mirrored[:, 0] = 2 * center_x - right_points_mirrored[:, 0]
        
        distances = np.linalg.norm(left_points - right_points_mirrored, axis=1)
        return 1.0 / (1.0 + np.mean(distances) / center_x)

    def calculate_face_coverage(self, landmarks: np.ndarray, shape: tuple) -> float:
        """Calculate face frame coverage ratio"""
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        width_ratio = (x_max - x_min) / shape[1]
        height_ratio = (y_max - y_min) / shape[0]
        return min(width_ratio, height_ratio)

    def save_template_visualization(self):
        """Save template with visualization overlays"""
        vis_img = (self.template_image * 255).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        # Draw face mask outline
        mask_edges = cv2.Canny(self.template_mask.astype(np.uint8), 100, 200)
        vis_img[mask_edges > 0] = [0, 255, 0]
        
        # Draw landmarks
        for pt in self.template_landmarks[self.FACE_CONTOUR_INDICES]:
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
            
        # Resize to 224x224
        vis_img_resized = cv2.resize(vis_img, (224, 224))
        cv2.imwrite(os.path.join(self.viz_dir, 'template_face.png'), vis_img_resized)

    def compute_contribution_map(self, 
                               image: torch.Tensor,
                               reference_pred: float,
                               face_mask: np.ndarray,
                               is_apex: bool = True,
                               other_image: torch.Tensor = None) -> np.ndarray:
        """Compute contribution map with consistent window size"""
        H, W = image.shape[-2:]
        window_size = self.window_size  # Using consistent window size
        stride = self.stride           # Using consistent stride

        contribution_map = np.zeros((H, W))
        count_map = np.zeros((H, W))
        
        # Ensure face mask is binary and float32
        face_mask = face_mask.astype(np.float32)
        face_mask[face_mask > 0] = 1.0
        
        for i in range(0, H - window_size + 1, stride):
            for j in range(0, W - window_size + 1, stride):
                window_mask = face_mask[i:i+window_size, j:j+window_size]
                if not np.any(window_mask):
                    continue
                    
                masked_img = image.clone()
                masked_img[:, i:i+window_size, j:j+window_size] = 0
                
                if is_apex:
                    pred_input = (other_image.unsqueeze(0).to(self.device), 
                                masked_img.unsqueeze(0).to(self.device))
                else:
                    pred_input = (masked_img.unsqueeze(0).to(self.device), 
                                other_image.unsqueeze(0).to(self.device))
                
                with torch.no_grad():
                    masked_pred = self.model(*pred_input, None)[1].squeeze().cpu().item()
                
                contribution = abs(masked_pred - reference_pred)
                contribution_map[i:i+window_size, j:j+window_size] += contribution
                count_map[i:i+window_size, j:j+window_size] += 1
        
        # Safely compute average contribution
        with np.errstate(divide='ignore', invalid='ignore'):
            contribution_map = np.divide(contribution_map, count_map, 
                                      where=count_map > 0, 
                                      out=np.zeros_like(contribution_map))
        
        # Apply face mask and remove any remaining invalid values
        masked_contribution = contribution_map * face_mask
        masked_contribution[np.isnan(masked_contribution)] = 0
        masked_contribution[np.isinf(masked_contribution)] = 0
        
        # Resize to 224x224
        masked_contribution_resized = cv2.resize(masked_contribution, (224, 224))
        
        return gaussian_filter(masked_contribution_resized, sigma=1)

    def get_alignment_transform(self, src_landmarks: np.ndarray, 
                              dst_landmarks: np.ndarray) -> np.ndarray:
        """Calculate face alignment transform"""
        src_points = src_landmarks[self.FACE_CONTOUR_INDICES]
        dst_points = dst_landmarks[self.FACE_CONTOUR_INDICES]
        matrix = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
        if matrix is None:
            matrix = np.eye(2, 3)
        return matrix

    def get_bdi_level(self, score: float) -> str:
        """Get depression severity level"""
        if score <= 13: return 'None (0-13)'
        elif score <= 19: return 'Mild (14-19)'
        elif score <= 28: return 'Moderate (20-28)'
        else: return 'Severe (29-63)'

    def analyze_sample(self, onset: torch.Tensor, 
                      apex: torch.Tensor, 
                      bdi_score: float) -> Optional[Dict]:
        """Analyze single sample with face region handling"""
        apex_np = apex.squeeze().numpy()
        landmarks = self.get_facial_landmarks(apex_np)
        if landmarks is None:
            return None
            
        face_mask = self.create_face_mask(landmarks, apex_np.shape)
        
        with torch.no_grad():
            onset_gpu = onset.to(self.device)
            apex_gpu = apex.to(self.device)
            original_pred = self.model(
                onset_gpu.unsqueeze(0),
                apex_gpu.unsqueeze(0),
                None
            )[1].squeeze().cpu().item()
        
        # Compute contribution maps
        apex_map = self.compute_contribution_map(
            apex_gpu, original_pred, face_mask,
            is_apex=True, other_image=onset_gpu
        )
        
        motion_pattern = torch.abs(apex_gpu - onset_gpu)
        dynamic_map = self.compute_contribution_map(
            motion_pattern, original_pred, face_mask,
            is_apex=False, other_image=motion_pattern
        )
        
        # Align to template
        matrix = self.get_alignment_transform(landmarks, self.template_landmarks)
        h, w = 224, 224  # Ensure output size is 224x224
        aligned_apex_map = cv2.warpAffine(apex_map, matrix, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
        aligned_dynamic_map = cv2.warpAffine(dynamic_map, matrix, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
        
        aligned_apex_map *= cv2.resize(self.template_mask, (224, 224))
        aligned_dynamic_map *= cv2.resize(self.template_mask, (224, 224))
        
        return {
            'apex_map': aligned_apex_map,
            'dynamic_map': aligned_dynamic_map,
            'bdi_score': bdi_score,
            'original_pred': original_pred
        }

    def analyze_dataset(self, dataloader: DataLoader) -> Dict:
        """Analyze dataset and collect results by severity level"""
        severity_maps = {
            'None (0-13)': {'apex': [], 'dynamic': []},
            'Mild (14-19)': {'apex': [], 'dynamic': []},
            'Moderate (20-28)': {'apex': [], 'dynamic': []},
            'Severe (29-63)': {'apex': [], 'dynamic': []}
        }
        
        print("\nAnalyzing facial patterns...")
        for batch_idx, (onset, apex, bdi_score, _) in enumerate(tqdm(dataloader)):
            onset, apex = onset.squeeze(0), apex.squeeze(0)
            bdi_score = bdi_score.item()
            
            result = self.analyze_sample(onset, apex, bdi_score)
            if result is not None:
                severity_level = self.get_bdi_level(bdi_score)
                severity_maps[severity_level]['apex'].append(result['apex_map'])
                severity_maps[severity_level]['dynamic'].append(result['dynamic_map'])
                
                # Save individual contribution maps
                np.savez_compressed(
                    os.path.join(self.maps_dir, f'sample_{batch_idx}_maps.npz'),
                    apex_map=result['apex_map'],
                    dynamic_map=result['dynamic_map'],
                    bdi_score=bdi_score,
                    prediction=result['original_pred']
                )
            
            if (batch_idx + 1) % 50 == 0:
                print("\nProgress Report:")
                for level, maps in severity_maps.items():
                    print(f"{level}: {len(maps['apex'])} samples")
        
        # Save severity distribution
        with open(os.path.join(self.viz_dir, 'severity_distribution.txt'), 'w') as f:
            f.write("Severity Distribution Summary\n")
            f.write("===========================\n\n")
            for level, maps in severity_maps.items():
                f.write(f"{level}: {len(maps['apex'])} samples\n")
        
        return severity_maps

    def visualize_overall_patterns(self, severity_maps: Dict):
        """Visualize patterns across all severity levels"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        feature_types = ['apex', 'dynamic']
        
        # Create enhanced background
        template_bg = cv2.normalize(
            self.template_image,
            None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        template_bg = cv2.cvtColor(template_bg, cv2.COLOR_GRAY2RGB)
        template_bg_resized = cv2.resize(template_bg, (224, 224))
        
        for feat_idx, feat_type in enumerate(feature_types):
            ax = axes[feat_idx]
            ax.imshow(template_bg_resized)
            
            overlay = np.zeros_like(template_bg_resized, dtype=np.float32)
            
            for severity_level, color in self.severity_colors.items():
                maps = severity_maps[severity_level][feat_type]
                if not maps:
                    continue
                
                normalized_maps = [m / (m.max() + 1e-10) for m in maps if m.max() > 0]
                if not normalized_maps:
                    continue
                
                mean_map = np.mean(normalized_maps, axis=0)
                threshold = np.percentile(mean_map[self.template_mask > 0], 80)
                significant_mask = (mean_map > threshold) & (self.template_mask > 0)
                
                for c in range(3):
                    overlay[:, :, c][significant_mask] = color[c]
                
                contours, _ = cv2.findContours(
                    significant_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    ax.plot(contour[:, 0, 0], contour[:, 0, 1],
                           color=color, linewidth=2, alpha=0.8)
            
            ax.imshow(overlay, alpha=0.5)
            ax.set_title(f'{feat_type.capitalize()} Features', fontsize=14)
            ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, facecolor=color, label=level)
            for level, color in self.severity_colors.items()
        ]
        fig.legend(handles=legend_elements, loc='center right',
                  bbox_to_anchor=(0.98, 0.5), title='Depression Severity')
        
        plt.suptitle('Overall Depression Pattern Distribution',
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'overall_patterns.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_severity_patterns(self, severity_maps: Dict):
        """Analyze and visualize patterns for each severity level"""
        for feature_type in ['apex', 'dynamic']:
            plt.figure(figsize=(20, 20))
            plt.suptitle(
                f'Depression Patterns Analysis - {feature_type.capitalize()} Features',
                fontsize=16, y=0.95
            )
            
            for idx, (severity, maps) in enumerate(severity_maps.items()):
                plt.subplot(2, 2, idx + 1)
                
                if len(maps[feature_type]) < 3:
                    plt.text(0.5, 0.5, f"{severity}\n(Insufficient samples: {len(maps[feature_type])})",
                           ha='center', va='center')
                    plt.axis('off')
                    continue
                
                self.visualize_severity_subpatterns(
                    maps[feature_type],
                    severity,
                    feature_type
                )
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(
                os.path.join(self.viz_dir, f'severity_patterns_{feature_type}.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close()

    def visualize_severity_subpatterns(self, 
                                 maps: List[np.ndarray],
                                 severity: str,
                                 feature_type: str,
                                 min_clusters: int = 1,
                                 max_clusters: int = 4):
        """Visualize pattern clusters within each severity level"""
        # Determine number of clusters
        n_clusters = min(max_clusters, max(min_clusters, len(maps) // 10))
        if n_clusters < 1:
            n_clusters = 1
        
        # Prepare data for clustering
        normalized_maps = []
        for m in maps:
            if np.any(m):
                m_norm = m / (m.max() + 1e-10)
                normalized_maps.append(m_norm.flatten())
        
        if len(normalized_maps) < n_clusters:
            n_clusters = max(2, len(normalized_maps))
        
        if not normalized_maps:
            plt.title(f"{severity}\n(No valid patterns)")
            plt.axis('off')
            return
            
        normalized_maps = np.stack(normalized_maps)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(normalized_maps)
        
        # Generate colors for patterns using updated colormap method
        pattern_colors = plt.colormaps['rainbow'].resampled(n_clusters)
        handled_patterns = set()
        
        # Visualize each cluster
        for cluster_idx in range(n_clusters):
            cluster_maps = np.array(maps)[clusters == cluster_idx]
            if len(cluster_maps) == 0:
                continue
            
            mean_pattern = np.mean(cluster_maps, axis=0)
            threshold = np.percentile(mean_pattern, 80)
            significant_mask = mean_pattern > threshold
            
            color = pattern_colors(cluster_idx)[:3]
            overlay = np.zeros((224, 224, 3))
            for c in range(3):
                overlay[:, :, c][significant_mask] = color[c]
            
            plt.imshow(overlay, alpha=0.5)
            
            contours, _ = cv2.findContours(
                significant_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                pattern_label = f'Pattern {cluster_idx+1}\n({len(cluster_maps)} samples)'
                for contour in contours:
                    plt.plot(contour[:, 0, 0], contour[:, 0, 1],
                            color=color, linewidth=2,
                            label=pattern_label if pattern_label not in handled_patterns else "")
                    handled_patterns.add(pattern_label)
        
        plt.title(f"{severity}\n{feature_type.capitalize()} Features\n({len(maps)} total samples)",
                fontsize=12)
        plt.axis('off')
        
        if handled_patterns:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    def run_analysis(self, dataloader: DataLoader):
        """Run complete analysis pipeline"""
        print("\nStarting depression pattern analysis...")
        
        # Select template and analyze dataset
        self.select_template(dataloader)
        severity_maps = self.analyze_dataset(dataloader)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_overall_patterns(severity_maps)
        self.analyze_severity_patterns(severity_maps)
        
        print(f"\nAnalysis complete! Results saved in: {self.save_dir}")

def main():
    """Main execution function"""
    # Set device and parameters
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    params = {
        'model_path': './Stage2_Final/stage_two_2014_Red_original_huber_13.0_20241128_131000/best_model.pth',
        'save_dir': './depression_analysis_results',
        'device': device
    }
    
    try:
        # Initialize analyzer
        analyzer = DepressionAnalyzer(**params)
        
        # Create dataset and dataloader
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
        
        # Run analysis
        analyzer.run_analysis(test_loader)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main()

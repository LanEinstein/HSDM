import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import random
import logging
import csv
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib import cm

def set_global_seed(seed=42):
    """
    Sets the global random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(save_path):
    """
    Configures a logger to record training information to both file and console.
    """
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(save_path, f'training_{timestamp}.log')
    
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

class ColorTransform:
    """
    Handles color mode transformations (e.g., RGB, Grayscale, etc.) 
    and normalizes the resulting image tensors.
    """
    def __init__(self, color_mode):
        self.color_mode = color_mode
        if color_mode == 'RGB':
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.out_channels = 3
        elif color_mode in ['Red', 'Green', 'Blue', 'Grayscale']:
            self.mean = [0.485]
            self.std = [0.229]
            self.out_channels = 1
        elif color_mode in ['RedGreen', 'RedBlue', 'GreenBlue']:
            # These modes expand to 2-channel images; a zero channel may be added later.
            self.mean = [0.485, 0.456, 0.0]
            self.std = [0.229, 0.224, 1.0]
            self.out_channels = 3
        else:
            raise ValueError(f"Unsupported color mode: {color_mode}")

    def __call__(self, img):
        img = self.color_transform(img)
        img = img.resize((224, 224))
        img = transforms.ToTensor()(img)
        
        # If the image has only 2 channels, add a zero channel to make it 3 channels.
        if self.out_channels == 3 and img.size(0) == 2:
            zero_channel = torch.zeros_like(img[0]).unsqueeze(0)
            img = torch.cat([img, zero_channel], dim=0)
            
        img = transforms.Normalize(mean=self.mean, std=self.std)(img)
        return img

    def color_transform(self, img):
        """
        Applies the requested color transformation (e.g., single channel, dual channel) to an RGB image.
        """
        img = img.convert('RGB')
        img_np = np.array(img)
        if self.color_mode == 'RGB':
            return img
        elif self.color_mode == 'Grayscale':
            return img.convert('L')
        elif self.color_mode in ['Red', 'Green', 'Blue']:
            channel_idx = {'Red': 0, 'Green': 1, 'Blue': 2}[self.color_mode]
            img_np = img_np[:, :, channel_idx]
            return Image.fromarray(img_np, mode='L')
        elif self.color_mode in ['RedGreen', 'RedBlue', 'GreenBlue']:
            channel_indices = {
                'RedGreen': [0, 1],
                'RedBlue': [0, 2],
                'GreenBlue': [1, 2]
            }[self.color_mode]
            img_np = img_np[:, :, channel_indices]
            # Convert to PIL Image
            if img_np.shape[2] == 2:
                return Image.fromarray(img_np)
            return Image.fromarray(img_np)
        else:
            raise ValueError(f"Unsupported color mode: {self.color_mode}")

class PairedTransform:
    """
    Applies the same transform to two images by setting the same random seed for both images.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img1 = self.transform(img1)
        random.seed(seed)
        torch.manual_seed(seed)
        img2 = self.transform(img2)
        return img1, img2

def get_transform(color_mode):
    """
    Returns a PairedTransform that applies a color transformation based on the specified color mode.
    """
    return PairedTransform(ColorTransform(color_mode))

class CheckpointManager:
    """
    Manages saving and loading of model checkpoints and best model weights.
    """
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.checkpoint_file = os.path.join(save_path, 'checkpoint.pth')
        self.best_model_file = os.path.join(save_path, 'best_model.pth')
        
    def save_checkpoint(self, epoch, model, optimizer, scheduler, best_rmse, best_epoch, results):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_rmse': best_rmse,
            'best_epoch': best_epoch,
            'results': results
        }, self.checkpoint_file)
        
    def save_best_model(self, model, optimizer, epoch, rmse):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rmse': rmse
        }, self.best_model_file)
        
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            return torch.load(self.checkpoint_file)
        return None

def create_id_mapping(train_csv, test_csv):
    """
    Creates a mapping from original ID to a new index for IDs present in both train and test sets.
    """
    train_ids = set()
    test_ids = set()
    
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            train_ids.add(int(row[3]))
    
    with open(test_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            test_ids.add(int(row[3]))
    
    common_ids = train_ids.intersection(test_ids)
    print(f"Train set has {len(train_ids)} unique IDs")
    print(f"Test set has {len(test_ids)} unique IDs")
    print(f"Common IDs between train and test: {len(common_ids)}")
    
    id_list = sorted(list(common_ids))
    id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(id_list)}
    
    return id_mapping

class PairedDataset(torch.utils.data.Dataset):
    """
    Custom dataset that loads pairs of images (onset and apex) with corresponding label and ID.
    """
    def __init__(self, csv_file, transform=None, id_mapping=None):
        self.data_pairs = []
        self.labels = []
        self.ids = []
        self.transform = transform
        self.max_label = 0
        self.id_mapping = id_mapping
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                onset_path, apex_path, label, id_label = row[:4]
                id_label = int(id_label)
                
                if id_label in self.id_mapping:
                    new_id = self.id_mapping[id_label]
                    self.data_pairs.append((onset_path, apex_path))
                    self.labels.append(float(label))
                    self.ids.append(new_id)
                    if float(label) > self.max_label:
                        self.max_label = float(label)
        
        print(f"Dataset: {csv_file}")
        print(f"Loaded {len(self.labels)} samples after ID filtering")
        print(f"Dataset contains {len(set(self.ids))} unique IDs")
        print(f"Max label value: {self.max_label}")
    
    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        onset_path, apex_path = self.data_pairs[idx]
        label = self.labels[idx]
        id_label = self.ids[idx]

        onset_image = Image.open(onset_path).convert('RGB')
        apex_image = Image.open(apex_path).convert('RGB')

        if self.transform:
            onset_image, apex_image = self.transform(onset_image, apex_image)
            
        onset_image = onset_image.float()
        apex_image = apex_image.float()
        label = torch.tensor(label, dtype=torch.float32)
        id_label = torch.tensor(id_label, dtype=torch.long)

        return onset_image, apex_image, label, id_label
    
    def get_max_label(self):
        return self.max_label
    
    def get_num_ids(self):
        return len(set(self.ids))

class CABlock(nn.Module):
    """
    A Convolution + Attention block that applies convolutional operations and a channel attention mechanism.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(CABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, groups=groups, bias=False)
        self.bn2 = norm_layer(planes)
        self.attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x):
        x, attn_last, if_attn = x
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        attn = torch.cat((avg_out, max_out), dim=1)
        attn = self.attn(attn)

        if attn_last is not None:
            attn = attn_last * attn

        attn = attn.repeat(1, self.planes, 1, 1)
        if if_attn:
            out = out * attn

        return out, attn[:, 0, :, :].unsqueeze(1), True

class ResNet(nn.Module):
    """
    A modified ResNet backbone that uses CABlock for attention-based convolutional layers.
    """
    def __init__(self, block, layers, in_channels=180, num_classes=1000, zero_init_residual=False,
                 groups=4, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple")
        
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False, groups=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = self._make_layer(block, 128, layers[0], groups=1)
        self.inplanes = int(self.inplanes * 1)
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], groups=1)
        self.inplanes = int(self.inplanes * 1)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], groups=1)
        self.inplanes = int(self.inplanes * 1)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], groups=1)
        self.inplanes = int(self.inplanes * 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, groups=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x, attn1, _ = self.layer1((x, None, True))
        temp = attn1
        attn1 = self.maxpool(attn1)

        x, attn2, _ = self.layer2((x, attn1, True))
        attn2 = self.maxpool(attn2)

        x, attn3, _ = self.layer3((x, attn2, True))
        attn3 = self.maxpool(attn3)

        x, attn4, _ = self.layer4((x, attn3, True))

        return x, temp

    def forward(self, x):
        return self._forward_impl(x)

def resnet18_pos_attention(in_channels=180):
    """
    Builds a ResNet with CABlock layers for position-based attention.
    """
    model = ResNet(CABlock, [1, 1, 1, 1], in_channels=in_channels)
    return model

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for lightweight feature extraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SqueezeExciteBlock(nn.Module):
    """
    Squeeze-and-Excitation block to adaptively recalibrate channel-wise responses.
    """
    def __init__(self, channel, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

class LightweightBranch(nn.Module):
    """
    A lightweight branch using depthwise separable convolutions for facial feature extraction.
    """
    def __init__(self, output_channels=512, input_channels=3):
        super(LightweightBranch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2),
            
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2),
            
            DepthwiseSeparableConv(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2),
            
            DepthwiseSeparableConv(256, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            
            SqueezeExciteBlock(output_channels),
        )
    
    def forward(self, x):
        return self.features(x)

class DiseaseRegressor(nn.Module):
    """
    A regressor to predict BDI-II scores from extracted features.
    """
    def __init__(self, feature_dim, max_label):
        super(DiseaseRegressor, self).__init__()
        self.max_label = max_label
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(24, 1),
        )
    
    def forward(self, x):
        out = self.regressor(x)
        return out

class ArcFaceLayer(nn.Module):
    """
    An ArcFace layer for identity classification with angular margin.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

class CenterLoss(nn.Module):
    """
    A center loss module that learns a center for each class to minimize intraclass distance.
    """
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        
    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels)
        loss = (features - centers_batch).pow(2).sum() / batch_size
        return loss

class IdentityClassifierV2(nn.Module):
    """
    An identity classifier that uses a fully connected feature extractor followed by an ArcFace layer.
    """
    def __init__(self, feature_dim, num_ids):
        super(IdentityClassifierV2, self).__init__()
        self.feature_dim = feature_dim
        self.num_ids = num_ids
        
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.arc_layer = ArcFaceLayer(128, num_ids, s=30.0, m=0.5)
        
    def forward(self, x, labels=None):
        features = self.features(x)
        if labels is not None:
            output = self.arc_layer(features, labels)
        else:
            output = F.linear(F.normalize(features), F.normalize(self.arc_layer.weight))
        return features, output

class DualAttentionFusion(nn.Module):
    """
    A fusion module that applies both spatial and channel attention to merge two feature maps.
    """
    def __init__(self, feature_dim=512, reduction=16):
        super().__init__()
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim // reduction, kernel_size=1),
            nn.BatchNorm2d(feature_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride=1, padding=3),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.channel_attention = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim // reduction, kernel_size=1),
            nn.BatchNorm2d(feature_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction, feature_dim * 2, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, motion_features, facial_features):
        concat_features = torch.cat([motion_features, facial_features], dim=1)
        
        spatial_attn = self.spatial_attention(concat_features)
        
        avg_pool = self.avg_pool(concat_features)
        max_pool = self.max_pool(concat_features)
        channel_attn = self.channel_attention(avg_pool + max_pool)
        
        attended_features = concat_features * spatial_attn * channel_attn
        refined_features = self.refine(attended_features)
        
        return refined_features

class EnhancedMMNetV2(nn.Module):
    """
    Main model that integrates motion features (onset-apex difference) and facial features 
    through dual attention fusion, then performs BDI-II regression and identity classification.
    """
    def __init__(self, num_ids=None, max_label=None, input_channels=3, device='cuda'):
        super().__init__()
        if max_label is None:
            raise ValueError("max_label must be specified")
            
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=180, kernel_size=3, 
                      stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(180),
            nn.ReLU(),
        )
        
        self.main_branch = resnet18_pos_attention(in_channels=180)
        
        self.lightweight_branch = LightweightBranch(output_channels=512, input_channels=input_channels)
        
        self.feature_fusion = DualAttentionFusion(feature_dim=512)
        
        self.disease_regressor = DiseaseRegressor(512, max_label)
        
        self.identity_classifier = IdentityClassifierV2(512, num_ids)
        
        self.center_loss = CenterLoss(num_ids, 128, device=device)

    def forward(self, onset, apex, id_label=None):
        motion_pattern = torch.abs(apex - onset)
        
        x = self.conv_act(motion_pattern)
        main_features, _ = self.main_branch(x)
        
        light_features = self.lightweight_branch(apex)
        
        combined_features = self.feature_fusion(main_features, light_features)
        
        bdi_pred = self.disease_regressor(combined_features)
        
        id_features, id_pred = self.identity_classifier(combined_features, id_label)
        
        return combined_features, bdi_pred, id_pred, id_features

def train_and_evaluate(color_mode, train_csv, test_csv, config, save_path, beta):
    """
    Trains and evaluates the EnhancedMMNetV2 model for a given color mode and HuberLoss delta (beta).
    """
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(save_path)
    logger.info(f"Training with beta value: {beta}, color mode: {color_mode}")
    
    checkpoint_manager = CheckpointManager(save_path)
    checkpoint = checkpoint_manager.load_checkpoint()
    start_epoch = 0
    best_rmse = float('inf')
    best_epoch = -1
    
    transform = get_transform(color_mode)
    id_mapping = create_id_mapping(train_csv, test_csv)
    train_dataset = PairedDataset(train_csv, transform=transform, id_mapping=id_mapping)
    test_dataset = PairedDataset(test_csv, transform=transform, id_mapping=id_mapping)
    max_label = max(train_dataset.get_max_label(), test_dataset.get_max_label())
    
    if color_mode == 'RGB' or color_mode in ['RedGreen', 'RedBlue', 'GreenBlue']:
        input_channels = 3
    else:
        input_channels = 1
        
    model = EnhancedMMNetV2(
        num_ids=len(id_mapping),
        max_label=max_label,
        input_channels=input_channels,
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    criterion = nn.HuberLoss(delta=beta).to(device)
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_rmse = checkpoint['best_rmse']
        best_epoch = checkpoint['best_epoch']
        logger.info(f"Resuming training from epoch {start_epoch}")
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=8,
        pin_memory=True,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=8,
        pin_memory=True,
        shuffle=False
    )
    
    epoch_rmse_list = []
    
    try:
        for epoch in range(start_epoch, config['num_epochs']):
            model.train()
            batch_losses = []
            
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Training]")
            for onset, apex, bdi_score, id_label in train_loader_tqdm:
                onset = onset.to(device, non_blocking=True)
                apex = apex.to(device, non_blocking=True)
                bdi_score = bdi_score.to(device, non_blocking=True)
                id_label = id_label.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                _, bdi_pred, _, _ = model(onset, apex, id_label)
                loss = criterion(bdi_pred.squeeze(), bdi_score)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                
            avg_loss = np.mean(batch_losses)
            logger.info(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
            
            scheduler.step()
            
            model.eval()
            predictions = []
            targets = []
            
            test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Evaluating]")
            with torch.no_grad():
                for onset, apex, bdi_score, _ in test_loader_tqdm:
                    onset = onset.to(device, non_blocking=True)
                    apex = apex.to(device, non_blocking=True)
                    _, bdi_pred, _, _ = model(onset, apex, None)
                    predictions.extend(bdi_pred.squeeze().cpu().numpy())
                    targets.extend(bdi_score.cpu().numpy())
            
            predictions = np.array(predictions)
            targets = np.array(targets)
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))
            epoch_rmse_list.append(rmse)
            logger.info(f"Epoch {epoch+1}, Test RMSE: {rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch
                checkpoint_manager.save_best_model(model, optimizer, epoch, rmse)
                logger.info(f"Best RMSE updated to {best_rmse:.4f} at epoch {epoch+1}")
            
            checkpoint_manager.save_checkpoint(
                epoch, model, optimizer, scheduler,
                best_rmse, best_epoch,
                {'rmse_list': epoch_rmse_list}
            )
            
    except Exception as e:
        logger.error(f"Training interrupted: {str(e)}")
        logger.info("Saving checkpoint before exit...")
        checkpoint_manager.save_checkpoint(
            epoch, model, optimizer, scheduler,
            best_rmse, best_epoch,
            {'rmse_list': epoch_rmse_list}
        )
        raise e
    
    logger.info(f"Training completed. Best RMSE: {best_rmse:.4f} at epoch {best_epoch+1}")
    return best_rmse

def plot_results(results, beta_values, color_modes):
    """
    Plots RMSE across different color modes and beta values for each dataset year.
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(14, 7))
    
    for year in ['2013', '2014']:
        num_betas = len(beta_values)
        colors = cm.cool(np.linspace(0, 1, num_betas)) if year == '2013' else cm.autumn(np.linspace(0, 1, num_betas))
        
        for idx, beta in enumerate(beta_values):
            rmses = [results[year][color_mode][beta] for color_mode in color_modes]
            plt.plot(color_modes, rmses, label=f'{year}, beta={beta}', color=colors[idx])
    
    plt.title('RMSE over Different Color Modes for Different Beta Values', fontsize=16)
    plt.xlabel('Color Mode', fontsize=14)
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('rmse_vs_color_modes.png', dpi=300)
    plt.close()

def main():
    """
    Entry point for training and evaluating the model on multiple settings of beta and color modes.
    """
    config = {
        'learning_rate': 4e-4,
        'num_epochs': 30,
        'batch_size': 128,
        'seed': 42,
    }
    
    beta_values = [1.78, 4.96, 8.0, 13.0, 16.0, 19.0, 24.0, 28.0, 30.0]
    color_modes = ['RGB', 'Grayscale', 'Red', 'Green', 'Blue', 'RedGreen', 'RedBlue', 'GreenBlue']
    
    datasets = {
        '2013': {
            'train_csv': 'BDI-II_train2013.csv',
            'test_csv': 'BDI-II_2013test.csv'
        },
        '2014': {
            'train_csv': 'BDI-II_train2014.csv',
            'test_csv': 'BDI-II_2014test.csv'
        }
    }
    
    results = {'2013': {}, '2014': {}}
    
    for year in ['2013', '2014']:
        results[year] = {}
        for color_mode in color_modes:
            results[year][color_mode] = {}
            for beta in beta_values:
                save_path = f'./results/{year}/{color_mode}/beta_{beta}'
                os.makedirs(save_path, exist_ok=True)
                
                best_rmse = train_and_evaluate(
                    color_mode=color_mode,
                    train_csv=datasets[year]['train_csv'],
                    test_csv=datasets[year]['test_csv'],
                    config=config,
                    save_path=save_path,
                    beta=beta
                )
                
                results[year][color_mode][beta] = best_rmse
    
    plot_results(results, beta_values, color_modes)

if __name__ == '__main__':
    main()

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime
import csv
from PIL import Image
from torchvision import transforms
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import seaborn as sns

# Set global random seed for reproducibility
def set_global_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
    
    return seed

# Get worker initialization function for DataLoader
def get_worker_init_fn(seed=42):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn

# Create ID mapping for the intersection of train and test dataset IDs
def create_id_mapping(train_csv, test_csv):
    train_ids = set()
    test_ids = set()
    
    # Get training set IDs
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            train_ids.add(int(row[3]))
    
    # Get test set IDs
    with open(test_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            test_ids.add(int(row[3]))
    
    # Get intersection of train and test IDs
    common_ids = train_ids.intersection(test_ids)
    print(f"Train set has {len(train_ids)} unique IDs")
    print(f"Test set has {len(test_ids)} unique IDs") 
    print(f"Common IDs between train and test: {len(common_ids)}")
    
    # Create mapping for common IDs only
    id_list = sorted(list(common_ids))
    id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(id_list)}
    
    return id_mapping

# Define dataset class for paired images
class PairedDataset(Dataset):
    def __init__(self, csv_file, transform=None, id_mapping=None):
        self.data_pairs = []
        self.labels = []
        self.ids = []
        self.transform = transform
        self.max_label = 0
        self.id_mapping = id_mapping
        
        # Read data and only keep samples with IDs in the mapping
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                onset_path, apex_path, label, id_label = row[:4]
                id_label = int(id_label)
                
                # Only include data if ID is in the mapping
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
            
        # Ensure the tensors are of type float32
        onset_image = onset_image.float()
        apex_image = apex_image.float()
        label = torch.tensor(label, dtype=torch.float32)
        id_label = torch.tensor(id_label, dtype=torch.long)

        return onset_image, apex_image, label, id_label
    
    def get_max_label(self):
        """Return the maximum label value."""
        return self.max_label
    
    def get_num_ids(self):
        """Return the number of unique IDs."""
        return len(set(self.ids))

# Setup training and test datasets with consistent ID mapping
def setup_datasets(train_csv, test_csv, transform):
    # Create a consistent ID mapping
    id_mapping = create_id_mapping(train_csv, test_csv)
    print(f"Total unique IDs: {len(id_mapping)}")
    
    # Create datasets
    train_dataset = PairedDataset(train_csv, transform=transform, id_mapping=id_mapping)
    test_dataset = PairedDataset(test_csv, transform=transform, id_mapping=id_mapping)
    
    return train_dataset, test_dataset, id_mapping

# Apply the same random transform to two images
class PairedTransform:
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

# Define the image transformations
def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return PairedTransform(transform)

# Define the CABlock module with attention mechanism
class CABlock(nn.Module):
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

# Define the ResNet model with positional attention
class ResNet(nn.Module):
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
                           "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                              bias=False, groups=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Main layers
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
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
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

        # Main branch processing
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

# Create a modified ResNet-18 model with positional attention
def resnet18_pos_attention(in_channels=180):
    model = ResNet(CABlock, [1, 1, 1, 1], in_channels=in_channels)
    return model

# Define Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
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

# Define Squeeze-and-Excitation block
class SqueezeExciteBlock(nn.Module):
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

# Define Lightweight Branch for facial feature extraction
class LightweightBranch(nn.Module):
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
            
            # Attention mechanism
            SqueezeExciteBlock(output_channels),
        )
    
    def forward(self, x):
        return self.features(x)

# Define Disease Regressor for BDI-II score prediction
class DiseaseRegressor(nn.Module):
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

# Define Dual Attention Fusion module
class DualAttentionFusion(nn.Module):
    def __init__(self, feature_dim=512, reduction=16):
        super().__init__()
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim // reduction, kernel_size=1),
            nn.BatchNorm2d(feature_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Channel Attention
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
        
        # Feature Refinement
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

# Define the Dynamic Branch Model
class DynamicBranchModel(nn.Module):
    def __init__(self, max_label):
        super().__init__()
        # Feature extraction network for motion patterns
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=180, kernel_size=3, 
                      stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(180),
            nn.ReLU(),
        )
        
        # Main branch for motion feature extraction
        self.main_branch = resnet18_pos_attention()
        
        # Disease Regressor
        self.disease_regressor = DiseaseRegressor(512, max_label)
    
    def forward(self, onset, apex):
        # Compute motion pattern
        motion_pattern = torch.abs(apex - onset)
        
        # Extract motion features
        x = self.conv_act(motion_pattern)
        main_features, _ = self.main_branch(x)
        
        # BDI-II regression
        bdi_pred = self.disease_regressor(main_features)
        
        return bdi_pred

# Define the Static Branch Model
class StaticBranchModel(nn.Module):
    def __init__(self, max_label):
        super().__init__()
        # Lightweight branch for facial feature extraction
        self.lightweight_branch = LightweightBranch(output_channels=512)
        
        # Disease Regressor
        self.disease_regressor = DiseaseRegressor(512, max_label)
    
    def forward(self, apex):
        # Extract facial features
        light_features = self.lightweight_branch(apex)
        
        # BDI-II regression
        bdi_pred = self.disease_regressor(light_features)
        
        return bdi_pred

# Define the Combined Model with dynamic and static branches
class CombinedModel(nn.Module):
    def __init__(self, max_label):
        super().__init__()
        # Feature extraction network for motion patterns
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=180, kernel_size=3, 
                      stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(180),
            nn.ReLU(),
        )
        
        # Main branch for motion feature extraction
        self.main_branch = resnet18_pos_attention()
        
        # Lightweight branch for facial feature extraction
        self.lightweight_branch = LightweightBranch(output_channels=512)
        
        # Dual attention fusion module
        self.feature_fusion = DualAttentionFusion(feature_dim=512)
        
        # Disease Regressor
        self.disease_regressor = DiseaseRegressor(512, max_label)
    
    def forward(self, onset, apex):
        # Compute motion pattern
        motion_pattern = torch.abs(apex - onset)
        
        # Extract motion features
        x = self.conv_act(motion_pattern)
        main_features, _ = self.main_branch(x)
        
        # Extract facial features
        light_features = self.lightweight_branch(apex)
        
        # Fuse features using dual attention
        combined_features = self.feature_fusion(main_features, light_features)
        
        # BDI-II regression
        bdi_pred = self.disease_regressor(combined_features)
        
        return bdi_pred

# Train and evaluate model with Huber Loss and RMSE metric
def train_and_evaluate(model, train_loader, test_loader, config, device, delta):
    criterion = nn.HuberLoss(delta=delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    
    best_rmse = float('inf')
    best_epoch = -1
    epoch_rmse_list = []
    
    for epoch in range(config['num_epochs']):
        model.train()
        batch_losses = []
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Training]", unit="batch")
    
        for data in train_loader_tqdm:
            if isinstance(model, DynamicBranchModel):
                onset, apex, bdi_score, _ = data
                onset = onset.to(device)
                apex = apex.to(device)
                bdi_score = bdi_score.float().to(device)
                optimizer.zero_grad()
                bdi_pred = model(onset, apex)
            elif isinstance(model, StaticBranchModel):
                _, apex, bdi_score, _ = data
                apex = apex.to(device)
                bdi_score = bdi_score.float().to(device)
                optimizer.zero_grad()
                bdi_pred = model(apex)
            elif isinstance(model, CombinedModel):
                onset, apex, bdi_score, _ = data
                onset = onset.to(device)
                apex = apex.to(device)
                bdi_score = bdi_score.float().to(device)
                optimizer.zero_grad()
                bdi_pred = model(onset, apex)
            else:
                raise ValueError("Unknown model type")
            
            loss = criterion(bdi_pred.squeeze(), bdi_score)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        avg_loss = np.mean(batch_losses)
        scheduler.step()
        
        # Evaluate on test set
        model.eval()
        predictions = []
        targets = []
    
        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Evaluating]", unit="batch")
    
        with torch.no_grad():
            for data in test_loader_tqdm:
                if isinstance(model, DynamicBranchModel):
                    onset, apex, bdi_score, _ = data
                    onset = onset.to(device)
                    apex = apex.to(device)
                    bdi_score = bdi_score.float().to(device)
                    bdi_pred = model(onset, apex)
                elif isinstance(model, StaticBranchModel):
                    _, apex, bdi_score, _ = data
                    apex = apex.to(device)
                    bdi_score = bdi_score.float().to(device)
                    bdi_pred = model(apex)
                elif isinstance(model, CombinedModel):
                    onset, apex, bdi_score, _ = data
                    onset = onset.to(device)
                    apex = apex.to(device)
                    bdi_score = bdi_score.float().to(device)
                    bdi_pred = model(onset, apex)
                else:
                    raise ValueError("Unknown model type")
                
                predictions.extend(bdi_pred.squeeze().cpu().numpy())
                targets.extend(bdi_score.cpu().numpy())
        
        # Calculate RMSE
        predictions = np.array(predictions)
        targets = np.array(targets)
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        epoch_rmse_list.append(rmse)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}")
    
    return best_rmse

# Plot RMSE comparison with highly distinguishable colors
def plot_results(results, beta_values, models):
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (16, 7),
    })

    # Use distinguishable colors
    distinct_colors = [
        '#E41A1C',  # Bright red
        '#377EB8',  # Dark blue
        '#4DAF4A',  # Green
        '#984EA3',  # Purple
        '#FF7F00',  # Orange
        '#FFFF33',  # Yellow
        '#A65628',  # Brown
        '#F781BF',  # Pink
        '#1B9E77'   # Cyan
    ]

    # Model markers
    model_markers = {
        'DynamicBranch': 'o',      # Circle
        'StaticBranch': 's',       # Square
        'Combined': '^'            # Triangle
    }
    marker_sizes = {'o': 100, 's': 90, '^': 120}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    def plot_dataset(ax, year):
        all_rmse = []
        for model in models:
            for beta in beta_values:
                all_rmse.append(results[year][model][beta])
        
        y_min, y_max = min(all_rmse), max(all_rmse)
        y_range = y_max - y_min
        
        y_min = y_min - y_range * 0.1
        y_max = y_max + y_range * 0.1
        
        # Plot data points and lines
        for i, beta in enumerate(beta_values):
            rmse_values = [results[year][model][beta] for model in models]
            x_positions = range(len(models))
            
            # Plot lines
            ax.plot(x_positions, rmse_values, 
                   color=distinct_colors[i], 
                   alpha=0.3,
                   linestyle='--', 
                   zorder=1,
                   linewidth=1.5)  # Increase line width
            
            # Plot data points
            for j, (model, rmse) in enumerate(zip(models, rmse_values)):
                ax.scatter(j, rmse,
                          marker=model_markers[model],
                          s=marker_sizes[model_markers[model]],
                          color=distinct_colors[i],
                          alpha=1.0,  # Fully opaque
                          zorder=3,
                          label=f'δ = {beta:.2f}' if j == 0 else "",
                          edgecolors='white',  # Add white edges
                          linewidth=1)  # Edge line width

        # Set axis and labels
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_title(f'Dataset {year}', pad=15, fontweight='bold')
        ax.set_xlabel('Model Type', labelpad=10)
        ax.set_ylabel('Root Mean Square Error (RMSE)', labelpad=10)
        
        # Set grid lines
        ax.grid(True, linestyle='--', alpha=0.2, zorder=0)
        
        # Optimize y-axis ticks
        yticks = np.linspace(y_min, y_max, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{y:.1f}' for y in yticks])

        # Add axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Plot two datasets
    plot_dataset(ax1, '2013')
    plot_dataset(ax2, '2014')

    # Create custom legend
    handles = []
    # δ value legend
    for i, beta in enumerate(beta_values):
        handles.append(plt.Line2D([0], [0], 
                                marker='o',
                                color=distinct_colors[i],
                                label=f'δ = {beta:.2f}',
                                markersize=8,
                                linestyle='none',
                                alpha=1.0,
                                markeredgecolor='white',
                                markeredgewidth=1))
    
    # Separator and model type title
    handles.append(plt.Line2D([0], [0], color='none', label=''))
    handles.append(plt.Line2D([0], [0], color='none', label='Model Types:'))
    
    # Model type legend
    for model in models:
        handles.append(plt.Line2D([0], [0],
                                marker=model_markers[model],
                                color='gray',
                                label=model,
                                markersize=10,
                                linestyle='none',
                                markeredgecolor='white',
                                markeredgewidth=1))

    # Legend position and style
    fig.legend(handles=handles,
              bbox_to_anchor=(1.02, 0.5),
              loc='center left',
              frameon=True,
              fancybox=True,
              shadow=True)

    # Main title
    fig.suptitle('Model Performance Comparison with Different Huber Loss δ Values',
                 fontsize=14,
                 fontweight='bold',
                 y=1.02)

    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_comparison_rmse.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.3)
    plt.close()

# Main function to run the training and evaluation
def main():
    config = {
        'learning_rate': 4e-4,
        'num_epochs': 30,
        'batch_size': 128,
        'seed': 42,
    }
    
    # Set delta values for Huber Loss
    deltas = [1.78, 4.96, 8.0, 13.0, 16.0, 19.0, 24.0, 28.0, 30.0]
    
    set_global_seed(config['seed'])
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
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
    
    models = ['DynamicBranch', 'StaticBranch', 'Combined']
    results = {'2013': {model: {} for model in models}, 
              '2014': {model: {} for model in models}}
    
    for year in ['2013', '2014']:
        train_csv = datasets[year]['train_csv']
        test_csv = datasets[year]['test_csv']
        
        transform = get_transform()
        train_dataset, test_dataset, id_mapping = setup_datasets(train_csv, test_csv, transform)
        max_label = max(train_dataset.get_max_label(), test_dataset.get_max_label())
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=get_worker_init_fn(config['seed'])
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=get_worker_init_fn(config['seed'])
        )
        
        for model_name in models:
            for delta in deltas:
                print(f"\nTraining {model_name} with delta={delta} on {year} dataset")
                
                if model_name == 'DynamicBranch':
                    model = DynamicBranchModel(max_label).to(device)
                elif model_name == 'StaticBranch':
                    model = StaticBranchModel(max_label).to(device)
                elif model_name == 'Combined':
                    model = CombinedModel(max_label).to(device)
                
                best_rmse = train_and_evaluate(model, train_loader, test_loader, 
                                             config, device, delta)
                results[year][model_name][delta] = best_rmse
                print(f"Year {year}, Model {model_name}, Delta {delta}, Best RMSE: {best_rmse:.4f}")
    
    # Plot results
    plot_results(results, deltas, models)

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn


class AvgMaxPool3d(nn.Module):
    """Concatena avg e max pool — dobra os canais de entrada do fc."""
    def __init__(self, output_size=2):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d(output_size)
        self.max = nn.AdaptiveMaxPool3d(output_size)

    def forward(self, x):
        a = self.avg(x).flatten(1)
        m = self.max(x).flatten(1)
        return torch.cat([a, m], dim=1)


# CNN BASELINE (volumes 3D)


class fMRICNN(nn.Module):
    def __init__(self, dropout: float, dataset: str):
        super().__init__()
        in_ch = 5 if dataset == "bold" else 1
        self.conv1 = nn.Conv3d(in_ch, 16, 7, stride=2)
        self.conv2 = nn.Conv3d(16, 32, 5, stride=2)
        self.conv3 = nn.Conv3d(32, 64, 3, stride=2)

        self.pool    = AvgMaxPool3d(output_size=2)  
        self.dropout = nn.Dropout3d(dropout)
        self.relu    = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 8, 256),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


class DeepfMRICNN(nn.Module):
    def __init__(self, dropout: float, dataset: str):
        super().__init__()
        in_ch = 5 if dataset == "bold" else 1
        self.conv1 = nn.Conv3d(in_ch, 16, 3, padding=1)
        self.features = nn.Sequential(
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
        self.pool = AvgMaxPool3d(output_size=2)  # 128*8*2 = 2048

        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 8, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.features(self.conv1(x))
        x = self.pool(x)
        x = self.fc(x)
        return x


class ShallowfMRICNN(nn.Module):
    def __init__(self, dropout_rate: float, dataset: str):
        super().__init__()
        in_ch = 5 if dataset == "bold" else 1
        self.conv1 = nn.Conv3d(in_ch, 16, 5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm3d(32)

        self.dropout = nn.Dropout3d(dropout_rate)
        self.pool    = AvgMaxPool3d(output_size=2)  # 32*8*2 = 512
        self.relu    = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(32 * 2 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.fc(x)
        return x



# MLP para vetores ROI flat (~13k-15k voxels)


class fMRIMLPSmall(nn.Module):
    def __init__(self, n_voxels: int, dropout: float, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_voxels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class fMRIMLPMedium(nn.Module):
    def __init__(self, n_voxels: int, dropout: float, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_voxels, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class fMRIMLPLarge(nn.Module):
    def __init__(self, n_voxels: int, dropout: float, n_classes: int = 3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_voxels, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.res_block1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.skip1 = nn.Linear(4096, 1024, bias=False)

        self.res_block2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.skip2 = nn.Linear(1024, 256, bias=False)

        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        x  = self.input_proj(x)
        x  = self.res_block1(x) + self.skip1(x)
        x  = self.res_block2(x) + self.skip2(x)
        return self.head(x)



# MedicalNet (volumes 3D)


class MedicalNetResNet(nn.Module):
    def __init__(self, medicalnet_model, n_classes=3, dataset="nsd", freeze_backbone=True):
        super().__init__()
        self.backbone = medicalnet_model

        in_ch = 5 if dataset == "bold" else 1
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv3d(
            in_ch, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )

        self.backbone.conv_seg = nn.Identity()

        # resnet10/18 tem 512 canais na última camada

        self.pool = AvgMaxPool3d(output_size=2) 

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# FACTORY

def create_model(model_name, dropout, dataset, cfg=None):
    """
    model_name:
        CNN volumétrico (3D):
            'cnn', 'deepcnn', 'shallowcnn', 'medicalnet'
        MLP para ROI flat:
            'mlp_small', 'mlp_medium', 'mlp_large'

    Para MLPs, cfg deve conter 'n_voxels' (int).
    """
    model_name = model_name.lower()
    dataset    = dataset.lower()

    # ── CNNs 3D ──────────────────────────────────────────────
    if model_name == "cnn":
        return fMRICNN(dropout, dataset)

    elif model_name == "deepcnn":
        return DeepfMRICNN(dropout, dataset)

    elif model_name == "shallowcnn":
        return ShallowfMRICNN(dropout, dataset)

    elif model_name == "medicalnet":
        if cfg is None or cfg.get("medicalnet_model") is None:
            raise ValueError("cfg['medicalnet_model'] deve ser fornecido.")
        return MedicalNetResNet(
            medicalnet_model=cfg["medicalnet_model"],
            dataset=dataset,
            freeze_backbone=cfg.get("freeze_backbone", True)
        )

    # ── MLPs ROI ─────────────────────────────────────────────
    elif model_name in ("mlp_small", "mlp_medium", "mlp_large"):
        if cfg is None or cfg.get("n_voxels") is None:
            raise ValueError(
                f"cfg['n_voxels'] deve ser fornecido para '{model_name}'. "
                f"Leia do HDF5: f['betas'].shape[1]"
            )
        n_voxels  = cfg["n_voxels"]
        n_classes = cfg.get("n_classes", 3)

        if model_name == "mlp_small":
            return fMRIMLPSmall(n_voxels, dropout, n_classes)
        elif model_name == "mlp_medium":
            return fMRIMLPMedium(n_voxels, dropout, n_classes)
        else:
            return fMRIMLPLarge(n_voxels, dropout, n_classes)

    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")
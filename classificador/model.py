import torch
import torch.nn as nn


# CNN BASELINE

class fMRICNN(nn.Module):

    def __init__(self, dropout: float, dataset: str): 
        super().__init__()

        # O bold inicializa com 5 canais (de TRs), enquanto NSD usa 1 canal
        if dataset == "bold":
            self.conv1 = nn.Conv3d(5,16,7,stride=2)
        else:
            self.conv1 = nn.Conv3d(1,16,7,stride=2)
        self.conv2 = nn.Conv3d(16,32,5,stride=2)
        self.conv3 = nn.Conv3d(32,64,3,stride=2)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(64,4)

        self.dropout = nn.Dropout3d(dropout)

        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = self.dropout(x)

        x = self.pool(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x

# DEEP CNN

class DeepfMRICNN(nn.Module):

    def __init__(self, dropout: float, dataset: str):
        super().__init__()

        if dataset == "bold":
            self.conv1 = nn.Conv3d(5, 16, 3, padding=1)
        else: 
            self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.features = nn.Sequential(

            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Dropout3d(dropout)
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        x = self.features(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ShallowfMRICNN(nn.Module):
    def __init__(self, dropout_rate: float, dataset: str):
        super().__init__()
        
        if dataset == "bold":
            self.conv1 = nn.Conv3d(5, 16, 5, stride=2, padding=2)
        else:
            self.conv1 = nn.Conv3d(1, 16, 5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm3d(16)
        
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        
        self.dropout = nn.Dropout3d(dropout_rate)
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Linear(32, 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x) 
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x

# ============================================================
# MEDICALNET WRAPPER
# ============================================================

# class MedicalNetResNet18(nn.Module):

#     def __init__(
#         self,
#         medicalnet_model,
#         n_classes=4,
#         freeze_backbone=True
#     ):
#         super().__init__()

#         self.backbone = medicalnet_model

#         # adapta primeira convolução
#         old_conv = self.backbone.conv1

#         self.backbone.conv1 = nn.Conv3d(
#             in_channels=5,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             padding=old_conv.padding,
#             bias=False
#         )

#         # adapta classificador final
#         in_features = self.backbone.fc.in_features

#         self.backbone.fc = nn.Linear(
#             in_features,
#             n_classes
#         )

#         # fine tuning parcial
#         if freeze_backbone:

#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#             for param in self.backbone.fc.parameters():
#                 param.requires_grad = True

#     def forward(self, x):

#         return self.backbone(x)


# ============================================================
# FACTORY
# ============================================================

def create_model(
    model_name,
    dropout,
    dataset
    # , medicalnet_model=None,
    # freeze_backbone=True
):

    model_name = model_name.lower()
    dataset = dataset.lower()

    if model_name == "cnn":

        return fMRICNN(dropout, dataset)

    elif model_name == "deepcnn":

        return DeepfMRICNN(dropout, dataset)
    
    elif model_name == "shallowcnn":

        return ShallowfMRICNN(dropout, dataset)

    # elif model_name == "medicalnet18":

    #     if medicalnet_model is None:

    #         raise ValueError(
    #             "medicalnet_model deve ser fornecido."
    #         )

    #     return MedicalNetResNet18(
    #         medicalnet_model=medicalnet_model,
    #         n_classes=n_classes,
    #         freeze_backbone=freeze_backbone
    #     )

    else:

        raise ValueError(f"Modelo '{model_name}' não suportado.")
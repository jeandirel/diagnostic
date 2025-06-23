import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    Implémentation d'une architecture U-Net pour la segmentation d'images.
    
    Le U-Net est composé d'un encodeur (partie contractante) qui capture le contexte
    et d'un décodeur (partie expansive) qui permet une localisation précise.
    Les connexions résiduelles (skip connections) préservent les détails spatiaux.
    
    Architecture:
        - Encodeur: 4 blocs avec MaxPooling (downsampling)
        - Bottleneck: bloc central avec les features les plus abstraites
        - Décodeur: 4 blocs avec ConvTranspose2d (upsampling) + skip connections
        - Sortie: convolution 1x1 pour classification par pixel
    
    Args:
        in_channels (int): Nombre de canaux d'entrée (défaut: 3 pour RGB)
        out_channels (int): Nombre de classes de sortie (défaut: 14)
        init_features (int): Nombre de features initial (défaut: 32)
    
    Example:
        >>> model = UNet(in_channels=3, out_channels=21, init_features=64)
        >>> x = torch.randn(2, 3, 512, 512)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([2, 21, 512, 512])
    """
    
    def __init__(self, in_channels=3, out_channels=14, init_features=32):
        super(UNet, self).__init__()
        features = init_features

        # ============= ENCODEUR (Partie contractante) =============
        # Chaque niveau d'encodage capture des features à différentes échelles
        
        # Niveau 1: features = 32, taille = H×W
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Réduction 2x

        # Niveau 2: features = 64, taille = H/2×W/2
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Niveau 3: features = 128, taille = H/4×W/4
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Niveau 4: features = 256, taille = H/8×W/8
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ============= BOTTLENECK (Goulot d'étranglement) =============
        # Point le plus profond: features = 512, taille = H/16×W/16
        # Capture les caractéristiques les plus abstraites et globales
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        # ============= DÉCODEUR (Partie expansive) =============
        # Chaque niveau remonte en résolution tout en affinant les prédictions
        
        # Remontée niveau 4: 512 → 256 features, H/16×W/16 → H/8×W/8
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        # Concaténation avec encoder4 (256+256=512) → 256 features
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        # Remontée niveau 3: 256 → 128 features, H/8×W/8 → H/4×W/4
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        # Concaténation avec encoder3 (128+128=256) → 128 features
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")

        # Remontée niveau 2: 128 → 64 features, H/4×W/4 → H/2×W/2
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        # Concaténation avec encoder2 (64+64=128) → 64 features
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        # Remontée niveau 1: 64 → 32 features, H/2×W/2 → H×W
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        # Concaténation avec encoder1 (32+32=64) → 32 features
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        # ============= COUCHE DE SORTIE =============
        # Convolution 1x1 pour mapper les features finales vers les classes
        # Pas d'activation car on attend des logits bruts pour CrossEntropyLoss
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        """
        Propagation avant du U-Net.
        
        Args:
            x (torch.Tensor): Tensor d'entrée de shape (batch_size, in_channels, H, W)
        
        Returns:
            torch.Tensor: Logits de sortie de shape (batch_size, out_channels, H, W)
        
        Étapes:
            1. Encodage avec réduction progressive de la résolution
            2. Bottleneck pour capturer le contexte global
            3. Décodage avec augmentation progressive + skip connections
            4. Classification finale par pixel
        """
        
        # ============= PHASE D'ENCODAGE =============
        # Sauvegarde des features pour les skip connections
        enc1 = self.encoder1(x)                    # Shape: (B, 32, H, W)
        enc2 = self.encoder2(self.pool1(enc1))     # Shape: (B, 64, H/2, W/2)
        enc3 = self.encoder3(self.pool2(enc2))     # Shape: (B, 128, H/4, W/4)
        enc4 = self.encoder4(self.pool3(enc3))     # Shape: (B, 256, H/8, W/8)

        # ============= BOTTLENECK =============
        bottleneck = self.bottleneck(self.pool4(enc4))  # Shape: (B, 512, H/16, W/16)

        # ============= PHASE DE DÉCODAGE =============
        # Décodage niveau 4: upsampling + skip connection + décodage
        dec4 = self.upconv4(bottleneck)           # Shape: (B, 256, H/8, W/8)
        dec4 = torch.cat((dec4, enc4), dim=1)     # Shape: (B, 512, H/8, W/8)
        dec4 = self.decoder4(dec4)                # Shape: (B, 256, H/8, W/8)

        # Décodage niveau 3
        dec3 = self.upconv3(dec4)                 # Shape: (B, 128, H/4, W/4)
        dec3 = torch.cat((dec3, enc3), dim=1)     # Shape: (B, 256, H/4, W/4)
        dec3 = self.decoder3(dec3)                # Shape: (B, 128, H/4, W/4)

        # Décodage niveau 2
        dec2 = self.upconv2(dec3)                 # Shape: (B, 64, H/2, W/2)
        dec2 = torch.cat((dec2, enc2), dim=1)     # Shape: (B, 128, H/2, W/2)
        dec2 = self.decoder2(dec2)                # Shape: (B, 64, H/2, W/2)

        # Décodage niveau 1
        dec1 = self.upconv1(dec2)                 # Shape: (B, 32, H, W)
        dec1 = torch.cat((dec1, enc1), dim=1)     # Shape: (B, 64, H, W)
        dec1 = self.decoder1(dec1)                # Shape: (B, 32, H, W)

        # ============= CLASSIFICATION FINALE =============
        # Pas d'activation; CrossEntropyLoss attend des logits bruts
        return self.conv(dec1)                    # Shape: (B, out_channels, H, W)

    @staticmethod
    def _block(in_channels, out_channels, name):
        """
        Crée un bloc convolutionnel standard du U-Net.
        
        Chaque bloc contient:
            - Conv2d 3x3 + BatchNorm + ReLU
            - Conv2d 3x3 + BatchNorm + ReLU
        
        Args:
            in_channels (int): Nombre de canaux d'entrée
            out_channels (int): Nombre de canaux de sortie
            name (str): Nom du bloc (pour debug/logging)
        
        Returns:
            nn.Sequential: Module séquentiel contenant les couches du bloc
            
        Notes:
            - padding=1 avec kernel_size=3 préserve la taille spatiale
            - bias=False car BatchNorm inclut un terme de biais
            - inplace=True pour économiser la mémoire
        """
        return nn.Sequential(
            # Première convolution du bloc
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=1,      # Préserve la taille: (3-1)/2 = 1
                bias=False      # BatchNorm gère le biais
            ),
            nn.BatchNorm2d(out_channels),           # Normalisation + stabilité
            nn.ReLU(inplace=True),                  # Activation non-linéaire
            
            # Seconde convolution du bloc
            nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


# ============= EXEMPLE D'UTILISATION =============
if __name__ == "__main__":
    print("=== Test du modèle U-Net ===")
    
    # Création du modèle avec paramètres par défaut
    model = UNet(in_channels=3, out_channels=14, init_features=32)
    
    # Calcul du nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Paramètres totaux: {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")
    
    # Test avec un batch d'images RGB 256x256
    print("\n=== Test de propagation avant ===")
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    print(f"Entrée: {x.shape}")
    
    # Propagation avant
    with torch.no_grad():  # Pas de calcul de gradients pour le test
        y = model(x)
    
    print(f"Sortie: {y.shape}")
    print(f"Expected: [{batch_size}, 14, 256, 256]")
    
    # Vérification des dimensions
    assert y.shape == (batch_size, 14, 256, 256), "Forme de sortie incorrecte!"
    print("✅ Test réussi! Les dimensions sont correctes.")
    
    # Test avec différentes tailles d'images
    print("\n=== Test avec différentes résolutions ===")
    test_sizes = [(128, 128), (512, 512), (64, 64)]
    
    for h, w in test_sizes:
        x_test = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y_test = model(x_test)
        print(f"Entrée {h}x{w} → Sortie {y_test.shape[2]}x{y_test.shape[3]}")
        assert y_test.shape[2:] == (h, w), f"Résolution incorrecte pour {h}x{w}!"
    
    print("✅ Tous les tests de résolution sont réussis!")
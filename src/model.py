import torch
import torch.nn as nn
from torchvision import models

def get_inception_for_finetuning(num_classes=32):
    # Carrega a rede pré-treinada no ImageNet
    # A sintaxe weights=... é o padrão nas versões recentes do PyTorch
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    
    # 1. Substituir a camada principal (fc = fully connected)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # 2. Substituir a camada auxiliar (AuxLogits)
    if model.AuxLogits is not None:
        aux_in_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)
        
    return model

if __name__ == "__main__":
    modelo = get_inception_for_finetuning(num_classes=32)
    print("Nova camada principal:", modelo.fc)
    print("Nova camada auxiliar:", modelo.AuxLogits.fc)
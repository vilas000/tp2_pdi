import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformações exigidas pelo artigo e pela Inception V3
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Range [-1, 1]
])

def get_source_dataloaders(data_dir="./dataset_split/source_domain", batch_size=32):
    train_dir = f"{data_dir}/train"
    test_dir = f"{data_dir}/test"
    
    # O ImageFolder automaticamente mapeia o nome da pasta para uma classe inteira (0 a 31)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    # Como você usa Linux, num_workers=4 agiliza bastante a leitura das imagens em background
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, train_dataset.classes

# Testando o loader
if __name__ == "__main__":
    train_loader, test_loader, classes = get_source_dataloaders()
    print(f"Total de classes no Source: {len(classes)}")
    
    # Pegando um batch para ver o formato (N, C, H, W)
    images, labels = next(iter(train_loader))
    print(f"Formato do batch de imagens: {images.shape}")
    print(f"Formato das labels: {labels.shape}")
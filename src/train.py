import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_source_dataloaders
from model import get_inception_for_finetuning

def train_source_domain(epochs=10, batch_size=32):
    # Detecta GPU (funciona tanto para CUDA quanto para ROCm da AMD)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando no dispositivo: {device}")

    # Carrega os dados
    print("Carregando datasets...")
    train_loader, test_loader, classes = get_source_dataloaders(batch_size=batch_size)
    
    # Inicia o modelo e joga para a placa de vídeo (ou CPU)
    model = get_inception_for_finetuning(num_classes=32).to(device)

    # O artigo cita o uso do Adam. A taxa de aprendizado 0.0001 é boa para fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # --- MODO DE TREINO ---
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # A Inception V3 retorna duas saídas no modo de treino
            outputs, aux_outputs = model(images)
            
            # Calcula a perda de ambas
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2 # 0.4 é o peso padrão para a saída auxiliar

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Printa o progresso a cada 100 batches
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Fim da Epoch {epoch+1} - Loss média de Treino: {running_loss/len(train_loader):.4f}")

        # --- MODO DE AVALIAÇÃO ---
        model.eval()
        correct = 0
        total = 0
        
        # Desliga o cálculo de gradientes para economizar memória e ficar mais rápido
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # No modo eval(), a Inception V3 retorna apenas a saída principal
                outputs = model(images)
                
                # Pega a classe com maior probabilidade
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100 * correct / total
        print(f"Acurácia no Teste (Source Domain): {accuracy:.2f}%\n")
        
    # Salva os pesos treinados para usarmos na Fase 2
    torch.save(model.state_dict(), "inception_source_trained.pth")
    print("Modelo salvo como 'inception_source_trained.pth'!")

if __name__ == "__main__":
    # Comece com poucas épocas só para garantir que o loop não vai estourar a memória (OOM)
    train_source_domain(epochs=5, batch_size=32)
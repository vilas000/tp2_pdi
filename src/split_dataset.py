import os
import random
import shutil
from pathlib import Path

def setup_dataset(original_data_dir, output_dir, target_classes_count=6, train_ratio=0.8, seed=42):
    # Fixar a seed para que o grupo todo tenha as mesmas classes alvo
    random.seed(seed)
    
    orig_path = Path(original_data_dir)
    out_path = Path(output_dir)
    
    # Pegar todas as pastas de classes (ignorando arquivos ocultos ou soltos)
    all_classes = [d.name for d in orig_path.iterdir() if d.is_dir()]
    all_classes.sort() # Ordenar antes de embaralhar garante consistência entre SOs diferentes
    
    if len(all_classes) != 38:
        print(f"Aviso: Esperava 38 classes, mas encontrei {len(all_classes)}.")

    # Sorteio dos domínios
    random.shuffle(all_classes)
    target_classes = all_classes[:target_classes_count]
    source_classes = all_classes[target_classes_count:]
    
    print(f"Classes Target ({len(target_classes)}): {target_classes}")
    
    # Função auxiliar para copiar e dividir
    def process_domain(classes, domain_name):
        for class_name in classes:
            class_path = orig_path / class_name
            images = [f for f in class_path.iterdir() if f.is_file()]
            random.shuffle(images)
            
            # Divisão 80/20
            split_idx = int(len(images) * train_ratio)
            train_imgs = images[:split_idx]
            test_imgs = images[split_idx:]
            
            # Criar os diretórios de destino
            train_out = out_path / domain_name / "train" / class_name
            test_out = out_path / domain_name / "test" / class_name
            train_out.mkdir(parents=True, exist_ok=True)
            test_out.mkdir(parents=True, exist_ok=True)
            
            # Copiar arquivos
            for img in train_imgs:
                shutil.copy2(img, train_out / img.name)
            for img in test_imgs:
                shutil.copy2(img, test_out / img.name)
                
            print(f"[{domain_name}] {class_name}: {len(train_imgs)} treino | {len(test_imgs)} teste")

    print("\nProcessando Source Domain...")
    process_domain(source_classes, "source_domain")
    
    print("\nProcessando Target Domain...")
    process_domain(target_classes, "target_domain")
    print("\nDivisão concluída com sucesso!")

# Substitua o caminho antigo por este:
ORIGINAL_DATA = "/home/vilas000/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage_dataset/color"
OUTPUT_DATA = "./dataset_split"

setup_dataset(ORIGINAL_DATA, OUTPUT_DATA)
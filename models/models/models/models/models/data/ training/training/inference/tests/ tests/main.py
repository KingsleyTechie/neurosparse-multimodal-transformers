import torch
from data.dataset import MultimodalDataset
from models.neurosparse import NeuroSparseTransformer
from training.trainer import NeuroSparseTrainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = MultimodalDataset(num_samples=800, seq_length=50, num_classes=5)
    val_dataset = MultimodalDataset(num_samples=200, seq_length=50, num_classes=5)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = NeuroSparseTransformer().to(device)
    
    trainer = NeuroSparseTrainer(model, device, learning_rate=1e-4)
    trainer.train(train_loader, val_loader, epochs=20)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

class NeuroSparseTrainer:
    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc='Training')
        for batch_idx, (visual, audio, imu, labels) in enumerate(progress_bar):
            visual = visual.to(self.device)
            audio = audio.to(self.device)
            imu = imu.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(visual, audio, imu)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc='Validation')
        with torch.no_grad():
            for visual, audio, imu, labels in progress_bar:
                visual = visual.to(self.device)
                audio = audio.to(self.device)
                imu = imu.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(visual, audio, imu)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                current_acc = 100. * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=20):
        print("Starting training...")
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 40)
            
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            val_loss, val_acc = self.validate(val_loader)
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            self.scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_neurosparse_model.pth')
                print(f'New best model saved with val_acc: {val_acc:.2f}%')

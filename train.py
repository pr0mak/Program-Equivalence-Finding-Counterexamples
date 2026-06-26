import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import CounterexampleMLP, VOCAB_SIZE
from dataset_loader import ProgramDataset

def train():
    print("Load data...")   
    dataset = ProgramDataset("straight_line_programs_10k_without_duplicates.csv")
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = CounterexampleMLP(vocab_size=VOCAB_SIZE)
    
    # loss function and optimizer / CrossEntropyLoss() includes softmax ?? 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # make adam optimizer lower and make more epochs
    
    # training loop
    epochs = 100
    best_val_loss = float('inf')
    
    print("Start training...\n")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch, _ in train_loader:
            optimizer.zero_grad()
            
            y_batch_10_vars = y_batch[:, :10].long()
            
            # forward pass: (BatchSize, 10, 2)
            predictions = model(X_batch)
            
            predictions_flat = predictions.view(-1, 2)     #predictions_flat = predictions.view(-1, 10) if 0-9
            y_batch_flat = y_batch[:, :10].flatten().long()

            loss = criterion(predictions_flat, y_batch_flat)
            
            # loss evaluation
            loss = criterion(predictions_flat, y_batch_flat)
            
            loss.backward()               
            optimizer.step()              
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                y_batch_10_vars = y_batch[:, :10].long()
                
                predictions = model(X_batch)
                
                predictions_flat = predictions.view(-1, 2) #predictions_flat = predictions.view(-1, 10) if 0-9
                y_batch_flat = y_batch_10_vars.reshape(-1)
                
                loss = criterion(predictions_flat, y_batch_flat)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # save best model(epoch?)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_mlp_model.pth")
            
    print("\nTraining complete! Best model saved: 'best_mlp_model.pth'")
    
if __name__ == "__main__": 
    train()
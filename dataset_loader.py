import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader, random_split

TOKENS = ['<PAD>', '+', '*', '//'] + \
        [f'const_{i}' for i in range(2)] + \
        [f'x{i}' for i in range(10)] + \
        [f'v{i}' for i in range(10)]   #[f'const_{i}' for i in range(10)] if 0-9

TOKEN_TO_ID = {token: i for i, token in enumerate(TOKENS)} 

class ProgramDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        code_sequence = ast.literal_eval(row['Code'])
        encoded_code = []
        for line in code_sequence:
            encoded_line = [TOKEN_TO_ID[token] for token in line]
            encoded_code.append(encoded_line)

        X_tensor = torch.tensor(encoded_code, dtype=torch.long)
        
        label_str = str(row['Label_Counterexample'])
        
        if label_str == 'None' or label_str == 'nan':
            target = [0.0] * 10 + [1.0]
        else:
            target = list(ast.literal_eval(label_str)) + [0.0]
            
        y_tensor = torch.tensor(target, dtype=torch.float32)
        
        return X_tensor, y_tensor,row['Code'] 

if __name__ == "__main__":
    print("Loading Data...")
    dataset = ProgramDataset('straight_line_programs_10k_without_duplicates.csv')
    
    total_size = len(dataset)
    print(f"Total problems in dataset: {total_size}")
    
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size 
    
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    X_batch, y_batch = next(iter(train_loader))
    print("\n--- Successful Test ---")
    print(f"Input shape (X): {X_batch.shape} -> (Batch=32, Lines=10, Words=4)")
    print(f"Output shape (y): {y_batch.shape}  -> NOW (Batch=32, Variables=11)")
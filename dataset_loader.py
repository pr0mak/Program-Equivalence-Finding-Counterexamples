import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader

TOKENS = ['<PAD>', '+', '*', '//', 'const_0', 'const_1'] + \
        [f'x{i}' for i in range(10)] + \
        [f'v{i}' for i in range(10)]

TOKEN_TO_ID = {token: i for i, token in enumerate(TOKENS)} 

class ProgramDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.dropna(subset=['Label_Counterexample'])
        self.df = self.df[self.df['Label_Counterexample'] != 'None'].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        code_sequence = ast.literal_eval(row['Code'])
        label_tuple = ast.literal_eval(row['Label_Counterexample'])
        
        encoded_code = []
        for line in code_sequence:
            encoded_line = [TOKEN_TO_ID[token] for token in line]
            encoded_code.append(encoded_line)

        X_tensor = torch.tensor(encoded_code, dtype=torch.long)
        
        y_tensor = torch.tensor(label_tuple, dtype=torch.float32)
        
        return X_tensor, y_tensor

if __name__ == "__main__":

    print("Load data...")
    dataset = ProgramDataset('straight_line_programs.csv')
    print(f"Found {len(dataset)} programs with a counterexample.")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    X_batch, y_batch = next(iter(dataloader))
    
    print("\n--- Successful test! ---")
    print(f"Input shape (X): {X_batch.shape} -> (Batch=32, Lines=10, Words=4)")
    print(f"Output shape (y): {y_batch.shape}  -> (Batch=32, Variables=10)")
    
    print("\nExample of the first program (as numbers):")
    print(X_batch[0])
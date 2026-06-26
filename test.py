import torch
import time
import ast
from torch.utils.data import DataLoader, random_split
from model import CounterexampleMLP, VOCAB_SIZE
from dataset_loader import ProgramDataset

def simulate_program(code_str, guessed_bits):
    code_sequence = ast.literal_eval(code_str)
    
    memory = {f'x{i}': int(guessed_bits[i]) for i in range(10)}
    for i in range(10):
        memory[f'const_{i}'] = i
    
    last_var = None
    for op, left, right, dest in code_sequence:
        val_l = memory[left]
        val_r = memory[right]
        
        if op == '+':
            res = val_l + val_r
        elif op == '*':
            res = val_l * val_r
        elif op == '//':
            res = val_l // val_r if val_r != 0 else 0
            
        memory[dest] = res
        last_var = dest
        
    return memory[last_var] != 0

def evaluate_model():
    print("Load test data (Test Set)...")
    dataset = ProgramDataset("straight_line_programs_10k_without_duplicates.csv") 
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    _, _, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    #load best epoch
    model = CounterexampleMLP(vocab_size=VOCAB_SIZE)
    model.load_state_dict(torch.load("best_mlp_model.pth", weights_only=True))
    model.eval() # eval mode
    
    correct_programs = 0
    total_programs = 0
    
    print("\nStart evaluation...")
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for X_batch, y_batch, code_batch in test_loader:
            
            y_batch_10_vars = y_batch[:, :10].long()
            
            logits = model(X_batch)
            predictions = torch.argmax(logits, dim=2) # Shape: (Batch, 10)
            
            for i in range(len(X_batch)):
                pred_10_bits = predictions[i]
                true_10_bits = y_batch_10_vars[i]
                
                is_zero_equiv = (y_batch[i, 10] == 1.0)
                
                if is_zero_equiv:
                    if (pred_10_bits == true_10_bits).all():
                        correct_programs += 1
                else:
                    if simulate_program(code_batch[i], pred_10_bits):
                        correct_programs += 1
                        
            total_programs += y_batch.size(0)
            
    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000
    avg_mlp_time = total_time_ms / total_programs
    success_rate = (correct_programs / total_programs) * 100
    
    print("-" * 30)
    print(f"Total Test: {total_programs}")
    print(f"Correct guess: {correct_programs}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average MLP Time per program: {avg_mlp_time:.4f} ms")
    print("-" * 30)
    
def evaluate_random_baseline():
    print("\n--- Load test data for RANDOM BASELINE ---")
    dataset = ProgramDataset("straight_line_programs_10k_without_duplicates.csv")
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    _, _, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    correct_programs = 0
    total_programs = 0
    
    with torch.no_grad():
        for X_batch, y_batch, code_batch in test_loader:
            
            random_predictions = torch.randint(0, 2, (len(X_batch), 10)) # random_predictions = torch.randint(0, 10, (len(X_batch), 10)) if 0-9
            
            for i in range(len(X_batch)):
                rand_10_bits = random_predictions[i]
                true_10_bits = y_batch[i, :10].long()
                is_zero_equiv = (y_batch[i, 10] == 1.0)
                
                if is_zero_equiv:
                    if (rand_10_bits == true_10_bits).all():
                        correct_programs += 1
                else:
                    if simulate_program(code_batch[i], rand_10_bits):
                        correct_programs += 1
                        
            total_programs += y_batch.size(0)
            
    success_rate = (correct_programs / total_programs) * 100
    print("=" * 40)
    print("RANDOM GUESS BASELINE RESULTS")
    print(f"Total Test: {total_programs}")
    print(f"Correct random guesses: {correct_programs}")
    print(f"Random Success Rate: {success_rate:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    evaluate_model()
    evaluate_random_baseline()
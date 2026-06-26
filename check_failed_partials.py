import torch
import torch.nn.functional as F
import ast
import numpy as np
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
        if op == '+': res = val_l + val_r
        elif op == '*': res = val_l * val_r
        elif op == '//': res = val_l // val_r if val_r != 0 else 0
        memory[dest] = res
        last_var = dest
    return memory[last_var] != 0

def analyze_failed_predictions():
    print("Loading test data for bit-by-bit error analysis...")
    dataset = ProgramDataset("straight_line_programs_10k_without_duplicates.csv") 
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    _, _, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = CounterexampleMLP(vocab_size=VOCAB_SIZE)
    model.load_state_dict(torch.load("best_mlp_model.pth", weights_only=True))
    model.eval() 
    
    correct_bits_in_failed_progs = []
    distribution_counts = {i: 0 for i in range(11)}
    
    conf_for_truly_correct_bits = []
    conf_for_truly_incorrect_bits = []
    
    total_failed_programs = 0
    
    print("Running evaluation and isolating failed instances...")
    with torch.no_grad():
        for X_batch, y_batch, code_batch in test_loader:
            y_batch_10_vars = y_batch[:, :10].long()
            
            logits = model(X_batch)         
            probs = F.softmax(logits, dim=2)  
            predictions = torch.argmax(logits, dim=2)
            
            for i in range(len(X_batch)):
                pred_10_bits = predictions[i]
                true_10_bits = y_batch_10_vars[i]
                is_zero_equiv = (y_batch[i, 10] == 1.0)
                
                overall_correct = False
                if is_zero_equiv:
                    if (pred_10_bits == true_10_bits).all():
                        overall_correct = True
                else:
                    if simulate_program(code_batch[i], pred_10_bits):
                        overall_correct = True
                        
                if not overall_correct:
                    total_failed_programs += 1
                    
                    bit_matches = (pred_10_bits == true_10_bits).cpu().numpy()
                    num_correct_bits = np.sum(bit_matches)
                    
                    correct_bits_in_failed_progs.append(num_correct_bits)
                    distribution_counts[num_correct_bits] += 1
                    
                    prob_chosen = probs[i, torch.arange(10), pred_10_bits]
                    conf_bits = torch.abs(prob_chosen - 0.5).cpu().numpy()
                    
                    for bit_idx in range(10):
                        if bit_matches[bit_idx]:
                            conf_for_truly_correct_bits.append(conf_bits[bit_idx])
                        else:
                            conf_for_truly_incorrect_bits.append(conf_bits[bit_idx])

    print("\n" + "="*50)
    print("=== FAILED PROGRAMS: BIT-BY-BIT ANALYSIS ===")
    print(f"Total failed programs analyzed: {total_failed_programs}")
    print(f"Average correct bits per failed program: {np.mean(correct_bits_in_failed_progs):.2f} / 10")
    print("-" * 50)
    print("Distribution of correct bits in failed instances:")
    for bits, count in distribution_counts.items():
        percentage = (count / total_failed_programs) * 100
        if percentage > 0.1:
            print(f"  {bits} bits correct: {count} programs ({percentage:.2f}%)")
            
    print("-" * 50)
    print("Confidence Profile Divergence (Distance from 0.5):")
    print(f"  Avg Confidence on CORRECT bits (within failure): {np.mean(conf_for_truly_correct_bits):.4f}")
    print(f"  Avg Confidence on INCORRECT bits (within failure): {np.mean(conf_for_truly_incorrect_bits):.4f}")
    print("=" * 50)

if __name__ == "__main__":
    analyze_failed_predictions()
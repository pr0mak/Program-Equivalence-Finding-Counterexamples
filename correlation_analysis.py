### check what percentage of the not correct are correct or not -> put picture of a bar as stat

### check partials confidence of the MLP bit by bit 
import torch
import torch.nn.functional as F
import ast
import numpy as np
from scipy.stats import pearsonr, spearmanr 
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

def run_statistical_analysis():
    print("Loading test data for statistical analysis...")
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
    
    all_confidence_scores = []
    all_correctness_flags = []
    total_programs = 0
    
    print("Evaluating predictions and capturing confidence profiles...")
    with torch.no_grad():
        for X_batch, y_batch, code_batch in test_loader:
            logits = model(X_batch)         
            probs = F.softmax(logits, dim=2)  
            predictions = torch.argmax(logits, dim=2)
            
            for i in range(len(X_batch)):
                pred_10_bits = predictions[i]
                true_10_bits = y_batch[i, :10].long()
                is_zero_equiv = (y_batch[i, 10] == 1.0)
                
                is_correct = 0
                if is_zero_equiv:
                    if (pred_10_bits == true_10_bits).all():
                        is_correct = 1
                else:
                    if simulate_program(code_batch[i], pred_10_bits):
                        is_correct = 1
                
                prob_chosen = probs[i, torch.arange(10), pred_10_bits]
                conf_bits = torch.abs(prob_chosen - 0.5).cpu().numpy()
                
                all_confidence_scores.append(np.mean(conf_bits))
                all_correctness_flags.append(is_correct)
            total_programs += y_batch.size(0)
            
    all_confidence_scores = np.array(all_confidence_scores)
    all_correctness_flags = np.array(all_correctness_flags)

    r_coeff, p_val = pearsonr(all_confidence_scores, all_correctness_flags)
    r_spearman, p_spearman = spearmanr(all_confidence_scores, all_correctness_flags)
    
    high_conf_mask = all_confidence_scores > 0.4
    high_conf_acc = np.mean(all_correctness_flags[high_conf_mask]) * 100 if np.sum(high_conf_mask) > 0 else 0.0
    
    print("\n" + "="*40)
    print("=== STATISTICAL RESULTS ===")
    print(f"Pearson Correlation Coefficient (r): {r_coeff:.4f}")
    print(f"P-value: {p_val:.4e}")
    print("===========================")
    print(f"Spearman Correlation Coefficient (r): {r_spearman:.4f}")
    print(f"P-value: {p_spearman:.4e}")
    print("===========================")
    print(f"Accuracy at High Confidence (>90%): {high_conf_acc:.2f}%")
    print(f"High Confidence Programs: {np.sum(high_conf_mask)} / {total_programs}")
    print("="*40)

if __name__ == "__main__":
    run_statistical_analysis()
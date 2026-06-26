import torch
import torch.nn as nn

VOCAB_SIZE = 26     #34 if 0-9

class CounterexampleMLP(nn.Module):

    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=16, hidden_dim=128, output_dim=11, num_lines=10, tokens_per_line=4):
        super(CounterexampleMLP, self).__init__()  #for back propagation

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) #hexadecimal

        flattened_size = num_lines * tokens_per_line * embedding_dim #compute linear size

        self.mlp = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),       #128 neurons
            nn.ReLU(),                                  #positives stay the same and negatives become 0
            nn.Dropout(p=0.3), #in order not to have overfitting
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim // 2, 20),    # nn.Linear(hidden_dim // 2, 100) if 0-9
        )   
        
    def forward(self, x):
        batch_size = x.size(0)

        embedded = self.embedding(x)                # sumbola -> dianismata 
        flattened = embedded.view(batch_size, -1)   # ? isopedwnei 3 teleutaies diastaseis

        logits = self.mlp(flattened) 
        # reshaping
        # softmax (s1)
        # loss (s2)
        
        # possibly loss combines already s1 and s2
        
        # 20 values for each program
        # (BatchSize, 20) -> (Batchsize, 10, 2) 
        logits_reshaped = logits.view(batch_size, 10, 2)             # logits_reshaped = logits.view(batch_size, 10, 10) if 0-9

        return logits_reshaped

if __name__ == "__main__":
    print("Test")
    
    batch_size = 32
    seq_length = 10
    vocab_size = 30 
    
    #fake data to test
    dummy_input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_length, 4))
    print(f"Input X shape: {dummy_input.shape}")

    model = CounterexampleMLP(vocab_size=vocab_size, output_dim=11)
    
    output = model(dummy_input)
    print(f"Output y shape: {output.shape} -> Must be (32, 10, 2)")
    print("The model outputs 2 logits (for class '0' and class '1') for each of the 10 variables!")
    
        # CrossEntropy on each 2 
        # Softmax
        # loss CrossEntropyloss(true_counter_example_var_value, network_output_for_that_var) 
        
        
        
import random
import itertools
import pandas as pd

OPERATORS = ['+', '*', '//']


def generate_random_program(num_inputs=10, num_lines=8):
    
    available_vars = [f"x{i}" for i in range(num_inputs)]
    available_vars.extend(["const_0", "const_1"])
    
    code = []
    
    for i in range(num_lines):
        op = random.choice(OPERATORS)
        op1 = random.choice(available_vars)
        op2 = random.choice(available_vars)
        
        if op == '//' and op2 == "const_0":
            op2 = "const_1" 
            
        new_var = f"v{i}"
        
        code.append((op, op1, op2, new_var))
        available_vars.append(new_var)
        
    return code

def execute_program(code, inputs):
    memory = inputs.copy()
    memory["const_0"] = 0
    memory["const_1"] = 1
    
    for op, op1, op2, res in code:
        val1 = memory[op1]
        val2 = memory[op2]
        
        if op == '+':
            memory[res] = val1 + val2
        elif op == '*':
            memory[res] = val1 * val2
        elif op == '//':
            memory[res] = val1 // val2 if val2 != 0 else 0
    
    last_var = code[-1][3]
    return memory[last_var]


def find_counterexample(code, num_inputs=10):
    
    all_possible_inputs = list(itertools.product([0, 1], repeat=num_inputs))
    steps = 0
    for input_tuple in all_possible_inputs:
        steps += 1
        input_dict = {f"x{i}": val for i, val in enumerate(input_tuple)}
        result = execute_program(code, input_dict)

        if result != 0:
            return input_tuple, steps 

    return None, steps


if __name__ == "__main__":
    dataset = []
    
    NUM_PROGRAMS = 100 
    
    print(f"Creation of: {NUM_PROGRAMS} programs...")
    
    for i in range(NUM_PROGRAMS):
        prog = generate_random_program(num_inputs=10, num_lines=8)
        counter_ex, steps = find_counterexample(prog, num_inputs=10)
        
        dataset.append({
            "Program_ID": i,
            "Code": str(prog),
            "Label_Counterexample": str(counter_ex),
            "Search_Steps": steps
        })

    df = pd.DataFrame(dataset)
    df.to_csv("straight_line_programs.csv", index=False)
    print("Dataset Created Successfully! File 'straight_line_programs.csv' is ready.")
import random
import itertools
import pandas as pd
from z3 import *

OPERATORS = ['+', '*', '//']


def generate_random_program(num_inputs, num_lines):
    
    available_vars = [f"x{i}" for i in range(num_inputs)]
    available_vars.extend([f"const_{i}" for i in range(2)])  #10 for 0-9
    
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
    for i in range(10):
        memory[f"const_{i}"] = i
    
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
    
    # all_possible_inputs = list(itertools.product([0, 1], repeat=num_inputs))
    # steps = 0
    # for input_tuple in all_possible_inputs:
    #     steps += 1
    #     input_dict = {f"x{i}": val for i, val in enumerate(input_tuple)}
    #     result = execute_program(code, input_dict)

    #     if result != 0:
    #         return input_tuple, steps 

    # return None, steps
    s = Solver()
    
    x = [Int(f"x{i}") for i in range(num_inputs)]
    
    for i in range(num_inputs):
        s.add(x[i] >= 0 , x[i] <= 1)  #9 if 0-9
    
    z3_memory = {f"x{i}": x[i] for i in range(num_inputs)}
    for i in range(10):
        z3_memory[f"const_{i}"] = i
    
    for op, op1, op2, res in code:
        val1 = z3_memory[op1]
        val2 = z3_memory[op2]
        
        if op == '+':
            z3_memory[res] = val1 + val2
        elif op == '*':
            z3_memory[res] = val1 * val2
        elif op == '//':
            is_literal_zero = False
            if isinstance(val2, int):
                if val2 == 0:
                    is_literal_zero = True
            elif is_expr(val2):
                try:
                    if val2.as_long() == 0:
                        is_literal_zero = True
                except:
                    pass
            
            if is_literal_zero:
                z3_memory[res] = 0
            else:
                z3_memory[res] = If(val2 != 0, val1 / val2, 0)
    
    last_var = code[-1][3]
    s.add(z3_memory[last_var] != 0)
    
    if s.check() == sat:
        model = s.model()
        counter_ex = tuple(int(model[x[i]].as_long()) for i in range(num_inputs))
        return counter_ex, 1
    else:
        return None,1


if __name__ == "__main__":
    dataset = []
    seen_programs = set()
    
    NUM_PROGRAMS = 10000    #1000000

    print(f"Creation of: {NUM_PROGRAMS} UNIQUE programs...")
    
    programs_created = 0
    duplicates_prevented = 0
    
    while programs_created < NUM_PROGRAMS:
        prog = generate_random_program(10, 10)
        prog_str = str(prog)
        
        if prog_str in seen_programs:
            duplicates_prevented += 1
            continue
            
        seen_programs.add(prog_str)
        
        counter_ex, steps = find_counterexample(prog, 10)
        
        dataset.append({
            "Program_ID": programs_created,
            "Code": prog_str,
            "Label_Counterexample": str(counter_ex),
            "Search_Steps": steps
        })
        
        programs_created += 1
    
        if programs_created % 1000 == 0:
            print(f"Generated {programs_created}/{NUM_PROGRAMS} programs...")

    df = pd.DataFrame(dataset)
    df.to_csv("straight_line_programs_1mil.csv", index=False)
    
    print("\nDataset Created Successfully!")
    print(f"File 'straight_line_programs_1mil.csv' is ready.")
    print(f"Prevented {duplicates_prevented} duplicate programs during generation.")
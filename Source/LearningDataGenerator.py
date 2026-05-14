from Types import Coefficients, Parameters

from TestDataGenerator import generate_coefficients
from DifferentialSolver import solve_population_equation

def generate_learning_data(num_entries, t_max : float = 500):

    # Constants
    r = 0.01
    p = 0.3
    q = 0.3
    
    # Output
    res : list[tuple[int, Coefficients]] = []
    
    while (len(res) < num_entries):
        coefficients = generate_coefficients(num_entries)
        for c in coefficients:
            params = Parameters.from_coefficients(c, r, p, q)
            print(len(res))
            res_1, res_2 = solve_population_equation(params, t_max)
            res.append((0 if res_1 > res_2 else 1, c))
            
    return res
               
        
def write_learning_data(data : list[tuple[int, Coefficients]]):
    with open("Data/LearningData2.txt", 'w') as output:
        output.write("# Winner ID (0 or 1) | h1 | h2 | s1 | s2 | a1 | a2 | b1 | b2 | z1_0 | z2_0 \n")
        
        for d in data:
            output.write(f"{d[0]} | {d[1].h1} | {d[1].h2} | {d[1].s1} | {d[1].s2} | {d[1].a1} | {d[1].a2} | {d[1].b1} | {d[1].b2} | {d[1].z1_0} | {d[1].z2_0}\n")
        
# Script
if __name__  == "__main__":
    data = generate_learning_data(5000)
    write_learning_data(data)
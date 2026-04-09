from TestDataGenerator import generate_test_data
from DifferentialSolver import solve_population_equation, Parameters


def generate_learning_data(num_entries):
    
    # Constants
    r = 0.01
    p = 0.3
    q = 0.3
    
    with open("Data/LearningData.txt", 'w') as output:
        output.write("# Winner ID (0 or 1) | h1 | h2 | s1 | s2 | a1 | a2 | b1 | b2 | z1_0 | z2_0 \n")
        
        generate_test_data(num_entries * 2)
        test_data_file = open("Data/TestData.txt")
        test_data_lines = test_data_file.readlines()
        
        i = 0
        num_good_entries = 0
        while(num_good_entries < num_entries):
            i += 1
               
            line = test_data_lines[i] 
            if line.startswith('#'): continue
            
            # h1 | h2 | s1 | s2 | a1 | a2 | b1 | b2 | z1_0 | z2_0
            h1, h2, s1, s2, a1, a2, b1, b2, z1_0, z2_0 = [float(i) for i in line.split('|')]
            
            params = Parameters(h1, h2, s1, s2, a1, a2, b1, b2, z1_0, z2_0, r, p, q)
            try:
                res_1, res_2 = solve_population_equation(params)
                num_good_entries += 1
                output.write(f"{0 if res_1 > res_2 else 1} | {h1} | {h2} | {s1} | {s2} | {a1} | {a2} | {b1} | {b2} | {z1_0} | {z2_0}\n")
                
            except:
                pass
            
        test_data_file.close()
        
        
# Script
if __name__  == "__main__":
    generate_learning_data(1000)
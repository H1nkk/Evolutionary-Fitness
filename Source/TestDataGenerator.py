import numpy as np
from Types import Coefficients

def generate_coefficients(num_entries : int) -> list[Coefficients]:
        rand_h1 = np.random.rand(num_entries)
        rand_h2 = np.random.rand(num_entries)
        
        rand_s1 = np.random.rand(num_entries)
        rand_s2 = np.random.rand(num_entries)
        
        rand_a1 = np.random.rand(num_entries)
        rand_a2 = np.random.rand(num_entries)
        
        rand_b1 = np.random.rand(num_entries)
        rand_b2 = np.random.rand(num_entries)

        rand_z1_0 = np.random.rand(num_entries)
        rand_z2_0 = np.random.rand(num_entries)
        
        res = [Coefficients(rand_h1[i], rand_h2[i], 
                            rand_s1[i], rand_s2[i], 
                            rand_a1[i], rand_a2[i], 
                            rand_b1[i], rand_b2[i], 
                            rand_z1_0[i], rand_z2_0[i]) for i in range(num_entries)]
        
        return res


def write_test_data(data : list[Coefficients]):
    with open("Data/TestData.txt", 'w') as f:
        f.write("# h1 | h2 | s1 | s2 | a1 | a2 | b1 | b2 | z1_0 | z2_0\n")
        for d in data:
            line = f"{d.h1} | {d.h2} | {d.s1} | {d.s2} | {d.a1} | {d.a2} | {d.b1} | {d.b2} | {d.z1_0} | {d.z2_0}\n"
            f.write(line)
        
        
if __name__  == "__main__":
    data = generate_coefficients(1000)
    write_test_data(data)
import numpy as np

def generate_test_data(num_entries : int):
    
    with open("Data/TestData.txt", 'w') as f:
        
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


        
        f.write("# h1 | h2 | s1 | s2 | a1 | a2 | b1 | b2 | z1_0 | z2_0\n")
        for i in range(num_entries):
            line = f"{rand_h1[i]} | {rand_h2[i]} | {rand_s1[i]} | {rand_s2[i]} | {rand_a1[i]} | {rand_a2[i]} | {rand_b1[i]} | {rand_b2[i]} | {rand_z1_0[i]} | {rand_z2_0[i]}\n"
            f.write(line)
        
        
if __name__  == "__main__":
    generate_test_data(1000)
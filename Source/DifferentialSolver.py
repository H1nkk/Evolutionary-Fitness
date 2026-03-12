import scipy as sp
from dataclasses import dataclass

@dataclass
class Parameters:
    h1 : float 
    h2 : float
    s1 : float
    s2 : float
    a1 : float
    a2 : float
    b1 : float
    b2 : float
    r : float
    p : float
    q : float

global_iteration_counter = 0
max_iterations = 10_000

class MaxIterationsReached(Exception):
    pass

# Defines a sytem of equations that will be solved
def system(t, z, params : Parameters):
    
    # Iteration control
    global global_iteration_counter
    global max_iterations
    
    global_iteration_counter += 1
    if (global_iteration_counter >= max_iterations):
        raise MaxIterationsReached("Max solving iterations exeeded!")
    
    """
    System of differential equations
    
    Parameters:
    t : float - time
    z : array - [z1, z2]
    """
    z1, z2 = z
    
    # First equation: dz1/dt
    term1_z1 = params.r * z1
    term2_z1 = params.h1 * z1**2
    term3_z1 = params.s1 * z2 * z1 * (1 + params.b1 * z1 + params.a1 * z2)
    term4_z1 = params.s2 * z2 * z1 * (1 + params.b2 * z2 + params.a2 * z1)
    term5_z1 = z1 * (params.q * z1 + params.p * z2)
    
    dz1dt = term1_z1 + term2_z1 + term3_z1 - term4_z1 - term5_z1
    
    # Second equation: dz2/dt
    term1_z2 = params.r * z2
    term2_z2 = params.h2 * z2**2
    term3_z2 = params.s2 * z2 * z1 * (1 + params.b2 * z2 + params.a2 * z1)
    term4_z2 = params.s1 * z2 * z1 * (1 + params.b1 * z1 + params.a1 * z2)
    term5_z2 = z2 * (params.q * z1 + params.p * z2)
    
    dz2dt = term1_z2 + term2_z2 + term3_z2 - term4_z2 - term5_z2
    
    return [dz1dt, dz2dt]


# Solves the population evaluation system of equations
def solve_population_equation(params : Parameters) -> tuple[float, float]:
    
    global global_iteration_counter
    global_iteration_counter = 0
    
    # Set initial conditions
    z1_0 = 1.0  # initial value for z1
    z2_0 = 1.0  # initial value for z2
    z0 = [z1_0, z2_0]

    # Set time span
    t_start = 0
    t_end = 1
    t_span = (t_start, t_end)

    # Solve the system
    solution = sp.integrate.solve_ivp(
        lambda t, z: system(t, z, params),
        t_span, 
        z0,
        method='RK45',
        rtol=1e-5,
        atol=1e-6,
    )
    
    return (solution.y[0][-1], solution.y[1][-1])


# Script
if __name__  == "__main__":
    
    # Constants
    r = 0.01
    p = 0.3
    q = 0.3

    with open("Data/TestData.txt") as f:

        lines = f.readlines()
        i = 0
        
        failed = 0
        won_1 = 0
        won_2 = 0
        
        for line in lines:
            
            if line.startswith('#'): continue
            
            # h1 | h2 | s1 | s2 | a1 | a2 | b1 | b2
            h1, h2, s1, s2, a1, a2, b1, b2 = [float(i) for i in line.split('|')]
            
            params = Parameters(h1, h2, s1, s2, a1, a2, b1, b2, r, p, q)
            
            global_iteration_counter = 0
            try:
                res_1, res_2 = solve_population_equation(params)
                
                won_1 += res_1 > res_2
                won_2 += res_2 > res_1
                
                print(f"{i} : {res_1 > res_2}: {res_1}, {res_2}")
                
            except Exception as e:
                failed += 1
                print(f"{i} : {str(e)}")
            i += 1
                
        print(f"Won 1 : {won_1}, Won 2 : {won_2}, Failed : {failed}")
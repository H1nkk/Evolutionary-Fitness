from dataclasses import dataclass

@dataclass
class Coefficients:
    h1 : float
    h2 : float
    s1 : float
    s2 : float
    a1 : float
    a2 : float
    b1 : float
    b2 : float
    z1_0 : float
    z2_0 : float
    
    
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
    z1_0 : float
    z2_0 : float
    r : float
    p : float
    q : float
    
    @classmethod
    def from_coefficients(cls, coef: Coefficients, r_ : float, p_ : float, q_ : float):
        return cls(coef.h1, coef.h2, coef.s1, coef.s2, coef.a1, coef.a2, coef.b1, coef.b2, coef.z1_0, coef.z2_0, r_, p_, q_)
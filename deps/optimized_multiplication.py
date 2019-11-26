#taken from:
#https://pythonandr.com/2015/10/13/karatsuba-multiplication-algorithm-python-code/
def karatsuba(x: int,y: int) -> int:
    """Function to multiply 2 numbers in a more efficient manner than the grade school algorithm"""
    if len(str(x)) == 1 or len(str(y)) == 1:
        return x*y
    else:
        #calculates the size of the numbers
        n = max(len(str(x)),len(str(y)))
        nby2 = n / 2
        
        #split the digit sequences about the middle
        a = x / 10**(nby2) #high1
        b = x % 10**(nby2) #low1
        c = y / 10**(nby2) #high2
        d = y % 10**(nby2) #low2
        
        #3 calls made to numbers approximately half the size
        ac = karatsuba(a,c)
        bd = karatsuba(b,d)
        ad_plus_bc = karatsuba(a+b,c+d) - ac - bd
        
        # this little trick, writing n as 2*nby2 takes care of both even and odd n
        prod = ac * 10**(2*nby2) + (ad_plus_bc * 10**nby2) + bd

        return prod


#another implementation
from math import *

def karatsuba2(x: int,y: int) -> int:
    # Set B = 10
    B = 10
    
    # Recursion base case
    if x < 10 or y < 10:
        return x*y    
    
    # m set to be length of x or y, whichever is maximum
    # This can be done using logarithms with base 10 or alternatively,
    # m = max(len(str(x)), len(str(y)))
    # But such a method will be inefficient for very large n
    m = max(int(log10(x)+1), int(log10(y)+1))
    
    # check whether m is even. If odd, set m lower by 1
    if m % 2 != 0:
        m -= 1
    m_2 = int(m/2)
    
    a, b = divmod(x, B**m_2)
    c, d = divmod(y, B**m_2)
    
    ac = karatsuba(a,c)
    bd = karatsuba(b,d)
    ad_bc = karatsuba((a+b),(c+d)) - ac - bd
    
    return ((ac*(10**m)) + bd + ((ad_bc)*(10**m_2)))
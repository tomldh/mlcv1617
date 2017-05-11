import numpy as np
from scipy.optimize import linprog
import sys

# to return the energy of unary factor
def unary(vnum):
    
    if vnum == 0:
        return (0.1,0.1)
    
    elif vnum == 1:
        return (0.1,0.9)
    
    elif vnum == 2:
        return (0.9,0.1)
    
    else:
        print('Error: unrecognized input label for unary potential')
        sys.exit()

# to return energy of pairwise factor
def pairwise(vnum1, vnum2, beta):
    
    if vnum1 == 0 and vnum2 == 0:
        return 0.
    
    elif vnum1 == 0 and vnum2 == 1:
        return beta
    
    elif vnum1 == 1 and vnum2 == 0:
        return beta
    
    elif vnum1 == 1 and vnum2 == 1:
        return 0.
    
    else:
        print('Error: unrecognized input label for pairwise potential')
        sys.exit()
        

if __name__ == '__main__':
    
    # 1.0-attractive, -1.0-repulsive
    beta = 1.0
    
    # indicator variable indices
    # (v1, v2, state of v1, state of v2)
    # in the case of unary, v1=v2
    yidx = {(0,0,0,0): 0,
            (0,0,1,1): 1,
            (1,1,0,0): 2,
            (1,1,1,1): 3,
            (2,2,0,0): 4,
            (2,2,1,1): 5,
            (0,1,0,0): 6,
            (0,1,0,1): 7,
            (0,1,1,0): 8,
            (0,1,1,1): 9,
            (0,2,0,0): 10,
            (0,2,0,1): 11,
            (0,2,1,0): 12,
            (0,2,1,1): 13,
            (1,2,0,0): 14,
            (1,2,0,1): 15,
            (1,2,1,0): 16,
            (1,2,1,1): 17}
    
    C = np.zeros((18), dtype='float64') #coefficient matrix
    A = np.zeros((18,18), dtype='float64') #constraint matrix
    b = np.zeros((18), dtype='float64') #RHS of equality constraint
    
    # three variables
    var = [0, 1, 2]
    
    # fill in coefficient matrix C
    for i in var:
        #unary energy
        C[yidx[(i,i,0,0)]] = unary(i)[0]
        C[yidx[(i,i,1,1)]] = unary(i)[1]
        
        for j in var[i+1:]:
            #pairwise energy
            C[yidx[(i,j,0,0)]] = pairwise(0, 0, beta)
            C[yidx[(i,j,0,1)]] = pairwise(0, 1, beta)
            C[yidx[(i,j,1,0)]] = pairwise(1, 0, beta)
            C[yidx[(i,j,1,1)]] = pairwise(1, 1, beta)
    
    print('Matrix C')
    print(C)
    
    row = 0
    # fill in constraint matrix
    for i in var:
        
        #e.g. y0_0 + y0_1 = 1
        A[row,yidx[(i,i,0,0)]] = 1.
        A[row,yidx[(i,i,1,1)]] = 1.
        b[row] = 1.
        
        row += 1
        
        for j in var[i+1:]:
            # e.g. y0_0 = y01_00 + y01_01
            A[row, yidx[(i,i,0,0)]] = 1.
            A[row, yidx[(i,j,0,0)]] = -1.
            A[row, yidx[(i,j,0,1)]] = -1.
            
            row += 1
            
            # e.g. y0_1 = y01_10 + y01_11
            A[row, yidx[(i,i,1,1)]] = 1.
            A[row, yidx[(i,j,1,0)]] = -1.
            A[row, yidx[(i,j,1,1)]] = -1.
            
            row += 1
            
            # e.g. y1_0 = y01_00 + y01_10
            A[row, yidx[(j,j,0,0)]] = 1.
            A[row, yidx[(i,j,0,0)]] = -1.
            A[row, yidx[(i,j,1,0)]] = -1.
            
            row += 1
            
            # e.g. y1_1 = y01_01 + y01_11
            A[row, yidx[(j,j,1,1)]] = 1.
            A[row, yidx[(i,j,0,1)]] = -1.
            A[row, yidx[(i,j,1,1)]] = -1.
            
            row += 1
            
            # e.g. y01_00 + y01_01 + y01_10 + y01_11 = 1
            A[row, yidx[(i,j,0,0)]] = 1.
            A[row, yidx[(i,j,0,1)]] = 1.
            A[row, yidx[(i,j,1,0)]] = 1.
            A[row, yidx[(i,j,1,1)]] = 1.
            b[row] = 1.
            
            row += 1
    
    print('Matrix A')
    print(A)
    print('Matrix b')  
    print(b)
    
    print('\n')
    print('beta: ', beta)
    
    res = linprog(c=C, A_eq=A, b_eq=b, options={"disp": True})
    
    print(res)
        
        
        
        
    
    
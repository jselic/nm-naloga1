import numpy as np
from src.Matrices import Givens, SymTri

def test_multiplication(n=100,m=10,drange=(2,20)):
    test_multiplication_checks = []
    for _ in range(n):
        #generate a random SymTry matrix
        dim = np.random.randint(*drange)
        diagonal = np.random.randint(*drange,size=dim)
        side_diagonal = np.random.randint(*drange,size=dim-1)
        symtri = SymTri(diagonal,side_diagonal)
        for _ in range(m):
            random_matrix = np.random.randint(*drange,size=(dim,dim))
            random_vector = np.random.randint(*drange,size=(1,dim))[0]
            test_multiplication_checks.append(np.array_equal(symtri * random_matrix, symtri.elements @ random_matrix))
    print(all(test_multiplication_checks))

def test_decomposition(n=300,drange=(2,50)):
    test_qr_checks = []
    for _ in range(n):
        #generate a random SymTri matrix
        dim = np.random.randint(*drange)
        diagonal = np.random.randint(*drange,size=dim)
        side_diagonal = np.random.randint(*drange,size=dim-1)
        symtri = SymTri(diagonal,side_diagonal)
        Q,R = symtri.qr()
        test_qr_checks.append(np.allclose(symtri.elements,Q@R,rtol=1e-10,atol=1e-10))
    print(all(test_qr_checks)) 

def test_eigen(n=100,drange=(2,20)):
    test_eigen_checks = []
    for _ in range(n):
        #generate a random SymTry matrix
        dim = np.random.randint(*drange)
        diagonal = np.random.randint(*drange,size=dim)
        side_diagonal = np.random.randint(*drange,size=dim-1)
        symtri = SymTri(diagonal,side_diagonal)
        
        gt = sorted(np.linalg.eig(symtri.elements)[0])
        cc = sorted(symtri.eigen()[0])
        
        test_eigen_checks.append(np.allclose(gt,cc,atol=1e-1))
    print(all(test_eigen_checks))
import numpy as np

class SymTri:
    """
    Razred Tridiagonalnih simetričnih matrik - Symmetric Tridiagonal - SymTri.

    Attributes:
        main_diagonal (list): Glavna diagonala
        side_diagonal (list): Vrhnja in spodnja diagonala.
        elements (np.ndarray): Matrika prevedena v tip np.ndarray, za namene testiranja

    Note:
        Additional notes, remarks, or important information.
    """
    def __init__(self, main_diagonal,side_diagonal):
        """
        Razredni konstruktor.

        Args:
            main_diagonal (list): Glavna diagonala.
            side_diagonal (list): Vrhnja in spodnja diagonala.
        """
        assert len(main_diagonal)-1 == len(side_diagonal), "The side diagonal must be one element smaller than the main one!"
        first_row = [main_diagonal[0]]
        size = len(main_diagonal)
        self.main_diagonal = main_diagonal
        self.side_diagonal = side_diagonal
        self.reset_elements()
            
    def reset_elements(self):
        self.elements = np.diag(self.main_diagonal) + np.diag(self.side_diagonal,k=1) + np.diag(self.side_diagonal,k=-1)
        
    def get_index(self, row, column):
        if abs(column-row) > 1:
            return 0
        elif column == row:
            return self.main_diagonal[column]
        else:
            return self.side_diagonal[min(row,column)]
        
    def set_index(self, row, column, value, mirror=True):
        assert abs(column-row) <= 1, "Cannot set a value that would make the matrix not Tridiagonal!"
        if abs(column-row) == 1:
            assert mirror, "mirror=True must be set in order to manipulate the side diagonal!"
        if column == row:
            self.main_diagonal[column] = value
        elif abs(column-row) == 1 and mirror:
            self.side_diagonal[min(column,row)] = value
        self.reset_elements()
        
    def first_index(self):
        return 0,0
    
    def last_index(self):
        last = len(self.main_diagonal)-1
        return last, last
        
    def mul_vector_right(self, other):
        """
        Metoda, ki pomnoži matriko z vektorjem z desne

        Args:
            other (list): Vektor, ki ga množimo z matriko.
        
        Returns:
            np.array: Pomnožen vektor.
        """
        s = self.side_diagonal
        m = self.main_diagonal
        return np.array([
            (s[i-1]*other[i-1] if i-1>=0 else 0) + m[i]*other[i] + (s[i]*other[i+1] if i+1 < len(other) else 0)
            for i,_ in enumerate(other)
        ])
        
    def __mul__(self,other):
        """
        Metoda, ki redefinira operacijo '*' nad matriko in objekti, ki so bodisi tudi matrika, ali pa vektor z ustrezno dimenzijo

        Args:
           other (list,tuple,np.ndarray): Objekt, s katerim množimo matriko z desne
        
        Returns:
            np.ndarray: Pomnožen vektor/matrika.
        """
        s = self.side_diagonal
        m = self.main_diagonal
        if isinstance(other, (list,tuple)):
            return self.mul_vector_right(other)
        elif isinstance(other,np.ndarray):
            newmatrix = np.zeros((len(other),len(other)))
            for i in range(len(other)):
                column = other[:,i]
                newmatrix[:,i] = self.mul_vector_right(column)
                
            return newmatrix
        
    def qr(self):
        """
        Metoda, ki izvede QR razcep SymTri matrike

        Args:
        
        Returns:
            np.ndarray: Ortogonalna matrika Q
            np.ndarray: Zgornje trikotna (dvodiagonalna) matrika R
        """
        Q = np.eye(len(self.main_diagonal))
        R = self.elements.copy()
        for i in range(len(self.side_diagonal)):
            G = Givens(R,i)
            R = G.elements @ R
            R[np.abs(R) < 1e-12] = 0
            Q = G.elements @ Q
            
        return Q.T, R
    
    def eigen(self, max_iterations=1000, tol=1e-10):
        """
        Metoda, ki izračuna lastne vrednosti in lastne vektorje

        Args:
            max_iterations (int): Maksimalno število iteracij pred zaključkom metode
            tol (float): Toleranca, pri kateri prenehamo izvajati
        
        Returns:
            np.ndarray: Array lastnih vrednosti
            np.ndarray: Array lastnih vektorjev
        """
        Q, R = self.qr()
        A = R @ Q
        eigen = R.diagonal()
        evect = Q.copy()
        for _ in range(max_iterations):
            A_new = SymTri(A.diagonal(),A.diagonal(offset=1))
            Q_new, R_new = A_new.qr()
            new_eigen = R_new.diagonal()
            if np.allclose(new_eigen,eigen,atol=tol):
                break
            A = R_new @ Q_new
            eigen = new_eigen
            evect = evect @ Q_new
            
        
        return eigen, evect
    

class Givens:
    """
    Razred Givensovih matrik

    Attributes:
        matrix (np.ndarray): Elementi SymTri matrike - v matrični formi
        position (int): Označuje pozicijo rotacijske matrike v Givensovi. Giblje se od 0 do n-1, kjer je n dimenzija originalne matrike.
        theta (float): rotacijski kot
        elements (np.ndarray): Matrika sama
    """
    def __init__(self, matrix, position):
        self.matrix = matrix
        self.position = position
        self.dim = len(matrix)
        self.theta = self.calculate_theta()
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        self.rotation_matrix = np.array([[c,-s],[s,c]])
        identity_with_rotation = np.eye(self.dim)
        identity_with_rotation[position:position+2,position:position+2] = self.rotation_matrix
        self.elements = identity_with_rotation
            
    def calculate_theta(self):
        d = self.matrix.diagonal()
        s = self.matrix.diagonal(offset=-1)
        theta = np.arctan(-s[self.position]/d[self.position])
        return theta
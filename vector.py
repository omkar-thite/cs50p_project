import math
import sys

class Vector:
    def __init__(self, cords):
        self.cords = cords

        self.dim = len(cords)
        
        # length of vector
        sum = 0   
        for x in cords:
            sum = sum + (x**2)     
        self.length = math.sqrt(sum)    
    

    # Getter for cords
    @property
    def cords(self):
        return self._cords
    
    # Setter function for cords
    @cords.setter
    def cords(self, cords):
        for i in cords:
            if type(i) in [int, float]:
                continue
            else:
                raise ValueError("Invalid entries in vector!")

        self._cords = cords
    
    # Only for 2D vector       
    def get_phi(self):
        if self.dim == 2:
            # cords[0] = x, cords[1] = y, cords[2] = z
            return math.atan(self.cords[1] / self.cords[0])
        if self.dim == 3:
            return math.acos(self.cords[0] / self.length)
    
    # Only for 3D vector       
    def get_theta(self):
        if self.dim == 3:
            return math.acos(self.cords[2] / self.length)


    def __str__(self):
        return f"{self.cords}, r = {self.length:.2f}"
        

    def __add__(self, other: 'Vector') -> 'Vector':
        '''
        :param other: vector to add
        :type other: vector
        :return: Added vector
        :rtype: vector

        '''
        dim_1 = self.dim
        dim_2 = other.dim
        
        if dim_1 == dim_2:
            added_cords = [self.cords[i] + other.cords[i] for i in range(dim_1)]
            return Vector(added_cords)
        
        elif dim_1==0:
            return other
        elif dim_2 == 0:
            return self
        else: 
            raise ValueError("Can't add vectors of different dimension!")
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        '''
        :param other: vector to be subtract
        :type other: vector
        :return: Subtracted vector
        :rtype: vector

        '''
        dim_1 = self.dim
        dim_2 = other.dim
        
        if dim_1 == dim_2:
            
            subtract_cords = [self.cords[i] - other.cords[i] for i in range(dim_1)]
            return Vector(subtract_cords)
        
        elif dim_1==0:
            return other
        
        elif dim_2 == 0:
            return self
        
        else: 
            raise ValueError("Can't subtract vectors of different dimension!")
    

    @classmethod
    def dot(cls, a: 'Vector', b: 'Vector') -> float:
        '''
        :param a, b: dot product vectors
        :type a, b: vector
        :return: dot product 
        :rtype: float
        :raise ValueError: If dimensions of two vectors are not matched
        :raise ValueError : If error occurs in dot product operation

        '''
        dim_1 = a.dim
        dim_2 = b.dim
        
        if dim_1 == dim_2:
            try:
                return sum(a_cord * b_cord for a_cord, b_cord in zip(a.cords, b.cords))
            except ValueError:
                raise ValueError("Error in dot product operation")
        else:
            raise ValueError("Not same dimensions!")


    # 2x2 Determinant
    @classmethod
    def det(cls, a: 'Vector', b: 'Vector') -> float:
        return (a[0] * b[1]) - (a[1] * b[0])

    # Only 3D
    # Implemented in view of C language
    @classmethod
    def cross(cls, p: 'Vector', q: 'Vector') -> 'Vector':
        '''
        :param p, q: cross product vectors
        :type p, q: vector
        :return: cross product 
        :rtype: Vector
        :raise ValueError: If dimensions of two vectors not 3

        '''
        if p.dim == q.dim and p.dim == 3:
            
                a = p.cords
                b = q.cords
                # c will store crossed values
                c = []

                # Over the first row
                for i in range(3): 
                   # New trimmed lists for every iteration
                    trim_a = []
                    trim_b = []

                    # trim a and b
                    for j in range(3): 
                        if (i != j):     
                            trim_a.append(a[j])
                            trim_b.append(b[j])

                    c.append(((-1)**i) * Vector.det(trim_a, trim_b))

        else:
                raise ValueError("Vectors must have the 3 dimensions!")

        return Vector(c) 
           
                
    def __mul__(self, other) -> 'Vector':
        '''
        Multiplies Vector by scalar
        :param other: scaler to multiply
        :type other: int, float
        :return: scaler multipled vector 
        :rtype: Vector
        :raise ValueError: if scaler is not passed after * operator

        '''
        # scalar must be on right side of operator *
        if type(other) in [int, float]:
            multiplied = [cord * other for cord in self.cords]
            return Vector(multiplied)
        
        else:
            raise ValueError("Invalid operand passed to operator *")

    def normalise(self):
        '''
        :return: Normalised Vector 
        :rtype: Vector
        :raise ZeroDivisionError: If 0 vector passed

        '''
        if self.length != 0:
            return self * (1 / self.length)

        else :
            raise ZeroDivisionError("Cannot normalise zero vector!")

    # Get angle between two vectors
    @classmethod
    def get_angle(cls, vector1, vector2) -> float:
        if not (isinstance(vector1, Vector) and (isinstance(vector2, Vector))):
            return None
        
        product = cls.dot(vector1, vector2)
        cos = product / (vector1.length * vector2.length)

        # provision for rounding errors
        if cos > 1:
            cos = 1
        if cos < -1:
            cos = -1

        return math.acos(cos)


class Matrix():
    def __init__(self, vectors):
        '''
        Matrix is a collection of row vectors in order
        attributes:
        rows: list of row vectors in ascendig order
        m : number of rows
        n: number of columns

        '''
        # List of rows
        self.rows = []

        self.m = len(vectors)

        # Populate a matrix
        if vectors:
            self.n = vectors[0].dim

            for vector in vectors:
                if len(vector.cords) != self.n:
                    raise ValueError(f"{vector.cords} does not contain same number of entries a first vector!")

                self.rows.append(vector)
        else:
            self.n = 0

    # Create a matrix from list of row vectors
    @classmethod
    def get_matrix(cls) -> 'Matrix':
        '''
        driver method not meant to be part of application
        '''
        cords = []

        try:
            m, n = input("Enter dimensions in format rows<space>columns: ").split()
            m  = int(m)
            n = int(n)

            for i in range(m):
                row = input(f"row {i}: ").strip().split()
                tmp = []

                # Convert to floats        
                for j in range(n):
                    tmp.append(float(row[j]))

                cords.append(Vector(tmp))
        
        except (ValueError, IndexError, TypeError):
            print("Invalid input!")
            return None

        return Matrix(cords)


    def __add__(self, other: 'Matrix') -> 'Matrix':
        '''
        Adds two matrices

        :param other: Matrix to add
        :return: Added matrix
        :rtype: Matrix

        '''
        if self.m == 0:
            return other
        elif other.m == 0:
            return self
        else:
            added = [self_row + other_row for self_row, other_row in zip(self.rows, other.rows)]
            return (Matrix(added))

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        '''
        Subtracts other matrix from self matrix

        :param other: Matrix to subtract
        :return: Subtracted matrix
        :rtype: Matrix

        '''
        if self.m == 0 or other.m == 0:
            raise ValueError("Empty matrix incompatible with operation")

        sub = [self_row - other_row for self_row, other_row in zip(self.rows, other.rows)]
        return Matrix(sub)


    def __str__(self):
        '''
        returns matrix in string format eg.
        [ [1, 2, 3]
         [3, 2, 1]
         [7, 5, 0] ]
        '''
        string = "["
        for i in self.rows:
            string += f"  {i.cords}\n"
        string = string.rstrip(string[-1])
        string += " ]\n"
        return string


    def __mul__(self, other: 'Matrix') -> 'Matrix':
        '''
        Multiples self matrix times other matrix
        Matrix multiplication is not implemented using standard algorithm but using abstaction of row vectors.
        Not efficient but good enough for application at hand


        :param other: Matrix to multiply
        :return: Multipled matrix
        :rtype: Matrix
        :raise ValueError: if dimensions of matrices are incompatible for multiplication

        '''

        # Scalar multiplication
        if(type(other) in [int, float]):
            
            product = [row * other for row in self.rows]
            return Matrix(product)
        
        # Matrix Multiplication
        elif (isinstance(other, Matrix)):
            
            if self.n != other.m:
                raise ValueError("Cannot multiply these matrices!")
            
            try:
                transpose = other.transpose()  
                c = [Vector([Vector.dot(self.rows[i], transpose.rows[j]) for j in range(other.m)]) for i in range(self.m)]
            except ValueError:
                raise ValueError("Error in dot product operation")
            else:
                return Matrix(c)
        
        else:
            raise ValueError("Invalid Matrix(s) in multiplying operation!")


    def transpose(self)->'Matrix':
        '''
        Transposes current matrix.

        :return: Transposed matrix
        :rtype: Matrix

        '''
        # Empty list to store rows as vectors
        transpose = [Vector([row.cords[i] for row in self.rows]) for i in range(self.n)]

        return Matrix(transpose)
    

    # Lower triangularise
    @classmethod
    def triangularise(matrix: 'Matrix') -> 'Matrix':
        '''
        This is a classmethod
        Convert a given matrix into a lower triangular matrix.

        :param matrix: Matrix to be triangularised
        :return: Modified lower triangularised matrix
        :rtype: Matrix
        :raise ValueError: if zero division occurs

        '''
        rows = matrix.rows
        m = matrix.m
        n = matrix.n

        for p in range(m):
            Matrix.pivoting(matrix, m, p)
            for i in range(p + 1, m):
                try:
                    factor = rows[i].cords[p] / rows[p].cords[p]
                    rows[i] = rows[i] - (rows[p] * factor)
                except ZeroDivisionError:
                    raise ValueError('Zero division error')
            
        return Matrix(rows)
    
    @classmethod
    def pivoting(matrix: 'Matrix', m: int, p: int) -> 'Matrix':
        '''
        This is a classmethod
        Convert a given matrix into a pivoted matrix.

        :param matrix: Matrix to be triangularised
        :param m: number of rows
        :param p: pivot row

        :return: Pivoted matrix
        :rtype: Matrix

        '''
        tmp = []
        b = []

        a = [row.cords for row in matrix.rows]

        max_row = p
        
        for i in range(p, m):
            if abs(a[i][p]) > abs(a[max_row][p]):
                max_row = i

        if (max_row != p):
            tmp = a[p] 
            a[p] = a[max_row]
            a[max_row] = tmp
        
        b = [Vector(row) for row in a]
        return Matrix(b)

    @classmethod
    def back(matrix: 'Matrix') -> Vector:
        '''
        This is a classmethod
        Convert a given matrix into a solution vector

        :param matrix: Matrix to be solved

        :return: solution vector
        :rtype: Vector
        '''
        rows = [vector.cords for vector in matrix.rows]

        m = matrix.m
        n = matrix.n
        x = [0 for _ in range(n - 1)]
        
        # Back substitution algorithm
        try:
            x[m - 1] = rows[m - 1][n - 1] / rows[m - 1][n - 2]

            for i in range(m - 2, -1, -1): # start from the second last row upto 0
                x[i] = rows[i][n - 1]
                for j in range(i + 1, n - 1):
                    x[i] = x[i] - rows[i][j] * x[j]

                x[i] = x[i]/rows[i][i]

        except ZeroDivisionError:
            return None
        return x


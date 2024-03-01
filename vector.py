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
        

    def __add__(self,other):
        dim_1 = self.dim
        dim_2 = other.dim
        
        if dim_1 == dim_2:
            
            added_cords = []
            for i in range(dim_1):
                added_cords.append(self.cords[i] + other.cords[i])
            return Vector(added_cords)
        elif dim_1==0:
            return other
        elif dim_2 == 0:
            return self
        else: 
            raise ValueError("Can't add vectors of different dimension!")
    
    def __sub__(self, other):
        dim_1 = self.dim
        dim_2 = other.dim
        
        if dim_1 == dim_2:
            
            subtract_cords = []
            
            for i in range(dim_1):
                subtract_cords.append(self.cords[i] - other.cords[i])
            return Vector(subtract_cords)
        
        elif dim_1==0:
            return other
        
        elif dim_2 == 0:
            return self
        
        else: 
            raise ValueError("Can't subtract vectors of different dimension!")
    
    @classmethod
    def dot(cls, a, b):
        dim_1 = a.dim
        dim_2 = b.dim
        
        if dim_1 == dim_2:
            sum = 0

            for i in range(dim_1):
                sum = sum + (a.cords[i] * b.cords[i])
            return sum
        else:
            raise ValueError("Not same dimensions!")

    # 2x2 Determinant
    @classmethod
    def det(cls, a, b):
        return (a[0] * b[1]) - (a[1] * b[0])

    # Only 3D
    # Implemented in view of C language
    @classmethod
    def cross(cls, p, q):
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
           
                
    def __mul__(self, other):
        # scalar must be on right side of operator *
        if type(other) in [int, float]:
            multiplied = []

            for i in range(0, self.dim):
                multiplied.append(self.cords[i] * other)
        
            return Vector(multiplied)
        else:
            raise ValueError("Invalid operand passed to operator *")

    def normalise(self):
            if self.length != 0:
                return self * (1 / self.length)

            else :
                raise ValueError("Cannot normalise zero vector!")

    # Get angle between two vectors
    @classmethod
    def get_angle(cls, vector1, vector2):
        if not (isinstance(vector1, Vector) and (isinstance(vector2, Vector))):
            print("Inputs are not vectors!")
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
    # vectors is list ov Vector objects i.e, rows of given matrix in order
    def __init__(self, vectors):
        # List of rows
        self.rows = []

        self.m = len(vectors)

        # Populate a matrix
        if vectors:
            self.n = vectors[0].dim

            for vector in vectors:
                if len(vector.cords) != self.n:
                    print(f"{vector.cords} does not contain same number of entries a first vector!")
                    return None
                self.rows.append(vector)
        else:
            self.n = 0

    # Create a matrix from list of row vectors
    @classmethod
    def get_matrix(cls):
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


    def __add__(self, other):
        if self.m == 0:
            return other
        elif other.m == 0:
            return self
        else:
            added = []
            for i in range(self.m):
                added.append(self.rows[i] + other.rows[i]) 
            return (Matrix(added))


    def __sub__(self, other):
        if self.m == 0:
            return other
        elif other.m == 0:
            return self

        sub = []
        for i in range(self.m):
            sub.append(self.rows[i] - other.rows[i]) 
        return Matrix(sub)


    def __str__(self):
        string = "["
        for i in self.rows:
            string += f"  {i.cords}\n"
        string = string.rstrip(string[-1])
        string += " ]\n"
        return string


    def __mul__(self,other):

        # Scalar multiplication
        if(type(other) in [int, float]):
            product = []
            for row in self.rows:
                product.append(row * other)
            return Matrix(product)
        
        # Multiplication of two matrices
        elif (type(other) == Matrix):
            if self.n != other.m:
                raise ValueError("Cannot multiply these matrices!")

            c = []
            transpose = other.transpose()
            
            for i in range(self.m):
                tmp = []
                for j in range(other.n):
                    try: 
                        tmp.append(Vector.dot(self.rows[i], transpose.rows[j]))
                    except ValueError:
                        sys.exit("Error in dot product operation!")
                c.append(Vector(tmp))
            
            return Matrix(c)
        else:

            raise ValueError("Invalid Matrix(ces) in multiplying operation!")


    def transpose(self):
        # Empty list to store rows as vectors
        transpose = []
        
        # Iterate over columns
        for i in range(self.n):    
            tmp = []
            # Iterate over rows 
            for row in self.rows:
                tmp.append(row.cords[i])
            
            # Matrix expects list of vectors 
            transpose.append(Vector(tmp))

        return Matrix(transpose)
    

    # Lower triangularise
    def triangularise(matrix):
        # matrix is Matrix
        rows = matrix.rows
        m = matrix.m
        n = matrix.n

        for p in range(m):
            Matrix.pivoting(matrix, m, p)
            for i in range(p + 1, m):
                factor = rows[i].cords[p] / rows[p].cords[p]
                rows[i] = rows[i] - (rows[p] * factor)
        
        return Matrix(rows)
    
        
    def pivoting(matrix, m, p):
        tmp = []
        a = []
        b = []

        for row in matrix.rows:
            a.append(row.cords)

        max_row = p
        
        for i in range(p, m):
            if abs(a[i][p]) > abs(a[max_row][p]):
                max_row = i

        if (max_row != p):
            tmp = a[p] 
            a[p] = a[max_row]
            a[max_row] = tmp
        
        for row in a:
            b.append(Vector(row))

        return Matrix(b)

    # Back substitution
    # returns list of ordered x values
    def back(matrix):
        rows = []
        for vector in matrix.rows:
            rows.append(vector.cords)

        m = matrix.m
        n = matrix.n
        x = []
        
        for i in range(n -1):
            x.append(0)

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


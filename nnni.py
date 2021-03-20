"""
Neural Networks with No Imports (in Python).
"""

MERSENNE_PRIME = 162259276829213363391578010288127
A = 2305843009213693951
B = 15485863

def random(seed):
    """Pseudo-random number generator in the range [0, 1).

    “Mersenne prime twister”."""

    state = seed
    for _ in range(20):
        state = (state*A + B) % MERSENNE_PRIME
    while True:
        state = (state*A + B) % MERSENNE_PRIME
        cand = state/(MERSENNE_PRIME + 1)
        # Ignore the first 3 decimal places.
        yield 1000*cand - int(1000*cand)

def ensure_other_is_scalar(matrix_method):
    """Simple decorator to check if second argument to a matrix method is a scalar."""
    def wrapper(self, other):
        if not isinstance(other, (int, float, complex)):
            raise ValueError(f"Cannot use {matrix_method} with 'other' of type {type(other)}.") 
        return matrix_method(self, other)
    return wrapper

class Matrix:
    """Represents a matrix with numerical components."""

    rand_generator = random(73)

    def __init__(self, data, nrows=None, ncols=None):
        """(nrows, ncols) gives the shape of the matrix and
        data populates the matrix."""

        self.nrows = nrows if nrows is not None else len(data)
        self.ncols = ncols if ncols is not None else len(data[0])

        if isinstance(data, (int, float, complex)):
            data = [[data for _ in range(self.ncols)] for _ in range(self.nrows)]
        self.data = data

    @ensure_other_is_scalar
    def __mul__(self, other):
        """Multiply a matrix with a scalar."""
        return self.map(lambda elem: elem*other)

    @ensure_other_is_scalar
    def __rmul__(self, other):
        return self*other

    @ensure_other_is_scalar
    def __lt__(self, other):
        return self.map(lambda elem: int(elem < other))

    @ensure_other_is_scalar
    def __le__(self, other):
        return self.map(lambda elem: int(elem <= other))

    @ensure_other_is_scalar
    def __eq__(self, other):
        return self.map(lambda elem: int(elem == other))

    @ensure_other_is_scalar
    def __ne__(self, other):
        return self.map(lambda elem: int(elem != other))

    @ensure_other_is_scalar
    def __gt__(self, other):
        return self.map(lambda elem: int(elem > other))

    @ensure_other_is_scalar
    def __ge__(self, other):
        return self.map(lambda elem: int(elem >= other))

    def map(self, f):
        """Map a function over all components of the matrix."""
        return Matrix([[f(elem) for elem in row] for row in self.data])

    @staticmethod
    def maximum(m1, m2):
        """Returns the component-wise maximum between two matrices."""

        if isinstance(m2, (int, float, complex)):
            return m1.map(lambda elem: max(elem, m2))

        data = []
        for row1, row2 in zip(m1.data, m2.data):
            data.append([max(e1, e2) for e1, e2 in zip(row1, row2)])
        return Matrix(data)

    @staticmethod
    def dot(m1, m2):
        """Perform matrix multiplication."""

        # Check if the shapes of the matrices are compatible.
        if m1.ncols != m2.nrows:
            raise ValueError(
                f"Cols of left matrix ({m1.ncols}) != rows of right matrix ({m2.nrows})."
            )
        # Compute the data of the resulting matrix.
        data = []
        for r in range(m1.nrows):
            row = []
            for c in range(m2.ncols):
                row.append(
                    sum(m1.data[r][i]*m2.data[i][c] for i in range(m1.ncols))
                )
            data.append(row)
        return Matrix(data)

    @staticmethod
    def random(nrows, ncols):
        """Generate a (nrows by ncols) random matrix.

        The values are drawn from the uniform distribution in [0, 1].
        """

        return Matrix(
            [[next(Matrix.rand_generator) for _ in range(ncols)] for _ in range(nrows)]
        )

class ActivationFunction:
    """'Abstract base class' for activation functions."""
    def loss(self, x):
        raise NotImplementedError("Activation functions should define the loss method.")

    def dloss(self, x):
        raise NotImplementedError("Activation functions should define the dloss method.")

class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def loss(self, x):
        return Matrix.maximum(x, self.alpha*x)

    def dloss(self, x):
        # return Matrix.maximum()
        return Matrix.maximum(x > 0, self.alpha)

class Layer:
    """An abstraction over a set of weights and biases between two sets of neurons."""
    def __init__(self, ins, outs):
        self.ins = ins
        self.outs = outs
        self.W = Matrix.random(outs, ins)
        self.b = Matrix.random(outs, 1)

class NeuralNetwork:
    """An ordered collection of compatible layers."""
    def __init__(self, layers):
        self.layers = layers

        # Check that the layers are compatible.
        for l1, l2 in zip(layers[::-1], layers[1::]):
            if l1.outs != l2.ins:
                raise ValueError(f"Layers are not compatible ({l1.outs} != {l2.ins}).")

if __name__ == "__main__":
    m1 = Matrix.random(2, 2)
    lrelu = LeakyReLU()
    print(lrelu.loss(m1).data)

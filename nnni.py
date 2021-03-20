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


class Layer:
    """An abstraction over a set of weights and biases between two sets of neurons."""
    def __init__(self, ins, outs):
        self.ins = ins
        self.outs = outs
        self.W = Matrix.random(outs, ins)
        self.b = Matrix.random(outs, 1)

if __name__ == "__main__":
    layer = Layer(16, 10)
    print(layer.W.data)
    print(layer.b.data)

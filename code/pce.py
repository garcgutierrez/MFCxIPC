"""
Orthonormal Hermite polynomial chaos basis and Galerkin algebra.

Conventions
-----------
* xi ~ N(0, I_d); psi_alpha(xi) = prod_i He_{alpha_i}(xi_i) / sqrt(alpha_i!)  (orthonormal).
* Total-degree truncation |alpha| <= p, graded ordering, psi_0 = 1.
* A random field u(xi) is stored as its coefficient vector u in R^P (last axis).
* Galerkin product (u * v)_k = sum_ij C_ijk u_i v_j,  C_ijk = E[psi_i psi_j psi_k]
  (fully symmetric because the basis is orthonormal).
* Multiplication matrix M_u with (M_u)_{kj} = sum_i C_ijk u_i, so u * v = M_u v.
"""
import itertools
import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


DENSE_C_MAX = 250          # above this basis size the dense triple-product tensor is not formed


def multi_indices(d, p):
    """Total-degree multi-indices, graded (degree, then reverse-lexicographic)."""
    out = []
    for deg in range(p + 1):
        block = [a for a in itertools.product(range(deg + 1), repeat=d) if sum(a) == deg]
        block.sort(reverse=True)
        out += block
    return np.array(out, dtype=int).reshape(-1, d)


def hermite_triple_1d(pmax):
    """T[a,b,c] = E[psi_a psi_b psi_c] for orthonormal probabilists' Hermite polynomials."""
    T = np.zeros((pmax + 1,) * 3)
    for a, b, c in itertools.product(range(pmax + 1), repeat=3):
        s2 = a + b + c
        if s2 % 2:
            continue
        s = s2 // 2
        if s < max(a, b, c):
            continue
        num = math.factorial(a) * math.factorial(b) * math.factorial(c)
        den = math.factorial(s - a) * math.factorial(s - b) * math.factorial(s - c)
        T[a, b, c] = num / den / math.sqrt(num)
    return T


def hermite_eval_1d(x, pmax):
    """Orthonormal He_n(x)/sqrt(n!) for n = 0..pmax; returns (len(x), pmax+1)."""
    x = np.asarray(x, dtype=float)
    H = np.zeros(x.shape + (pmax + 1,))
    H[..., 0] = 1.0
    if pmax >= 1:
        H[..., 1] = x
    for n in range(1, pmax):
        H[..., n + 1] = x * H[..., n] - n * H[..., n - 1]
    return H / np.sqrt([math.factorial(n) for n in range(pmax + 1)])


class PCBasis:
    def __init__(self, d, p):
        self.d, self.p = d, p
        self.alphas = multi_indices(d, p)
        self.P = len(self.alphas)
        self.degree = self.alphas.sum(axis=1)
        T = hermite_triple_1d(p)
        A = self.alphas
        from scipy.sparse import csr_matrix
        if self.P <= DENSE_C_MAX:
            C = np.ones((self.P,) * 3)
            for i in range(d):
                C *= T[A[:, None, None, i], A[None, :, None, i], A[None, None, :, i]]
            C[np.abs(C) < 1e-14] = 0.0
            self.C = C
            self.Cflat = C.reshape(self.P, self.P * self.P)
            I, J, K = np.nonzero(C)
            V = C[I, J, K]
        else:
            # large bases: build the non-zeros slice by slice without the dense P^3 tensor
            self.C = self.Cflat = None
            Is, Js, Ks, Vs = [], [], [], []
            for i in range(self.P):
                S = np.ones((self.P, self.P))
                for l in range(d):
                    S *= T[A[i, l], A[:, None, l], A[None, :, l]]
                jj, kk = np.nonzero(np.abs(S) >= 1e-14)
                Is.append(np.full(len(jj), i)); Js.append(jj); Ks.append(kk); Vs.append(S[jj, kk])
            I, J, K, V = (np.concatenate(a) for a in (Is, Js, Ks, Vs))
        self.nnz = int(len(I))
        # sparse (COO) form used for cheap batched products: out_k = sum V u_I v_J
        self._I, self._J = I, J
        self._S = csr_matrix((V, (np.arange(len(I)), K)), shape=(len(I), self.P))
        # scatter matrix for O(nnz) assembly of multiplication matrices: (M_u)_{jk} = sum_i C_ijk u_i
        self._MT = csr_matrix((V, (J * self.P + K, I)), shape=(self.P * self.P, self.P))
        self.e0 = np.zeros(self.P)
        self.e0[0] = 1.0
        # index of the first-order coefficient of each xi_i
        self.lin_index = [int(np.where((A == np.eye(d, dtype=int)[i]).all(1))[0][0]) for i in range(d)]

    # ---- Galerkin algebra -------------------------------------------------
    def mult_matrix(self, u):
        """M_u for u of shape (..., P) -> (..., P, P), assembled from the sparse tensor in O(nnz(C)) per
        input (the dense contraction u @ Cflat costs O(P^3) and is kept as mult_matrix_dense)."""
        u = np.asarray(u, float)
        u2 = u.reshape(-1, self.P)
        M = np.asarray(self._MT @ u2.T).T
        return M.reshape(u.shape[:-1] + (self.P, self.P))

    def mult_matrix_dense(self, u):
        u = np.asarray(u)
        return (u.reshape(-1, self.P) @ self.Cflat).reshape(u.shape[:-1] + (self.P, self.P))

    def prod(self, u, v):
        """Galerkin product, broadcasting over leading axes."""
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        sh = u.shape
        M = self.mult_matrix(u.reshape(-1, self.P))
        return np.einsum('bjk,bj->bk', M, v.reshape(-1, self.P)).reshape(sh)

    def sprod(self, u, v):
        """Sparse Galerkin product; cost O(batch * nnz). Broadcasts leading axes."""
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        sh = u.shape
        u2, v2 = u.reshape(-1, self.P), v.reshape(-1, self.P)
        nb = u2.shape[0]
        chunk = max(1, int(4e6 // max(len(self._I), 1)))      # bound the (batch x nnz) temporary
        if nb <= chunk:
            return np.asarray((u2[:, self._I] * v2[:, self._J]) @ self._S).reshape(sh)
        out = np.empty((nb, self.P))
        for i in range(0, nb, chunk):
            out[i:i + chunk] = (u2[i:i + chunk, self._I] * v2[i:i + chunk, self._J]) @ self._S
        return out.reshape(sh)

    def matvec(self, M, v):
        return np.einsum('...jk,...j->...k', M, v)

    # ---- evaluation / statistics -------------------------------------------
    def eval_basis(self, xi):
        """xi: (N, d) -> Psi (N, P)."""
        xi = np.atleast_2d(xi)
        H = hermite_eval_1d(xi, self.p)            # (N, d, p+1)
        Psi = np.ones((xi.shape[0], self.P))
        for i in range(self.d):
            Psi *= H[:, i, self.alphas[:, i]]
        return Psi

    def evaluate(self, coeffs, xi=None, Psi=None):
        """coeffs (..., P) evaluated at samples -> (..., N)."""
        if Psi is None:
            Psi = self.eval_basis(xi)
        return np.asarray(coeffs) @ Psi.T

    @staticmethod
    def mean(c):
        return np.asarray(c)[..., 0]

    @staticmethod
    def var(c):
        return np.sum(np.asarray(c)[..., 1:] ** 2, axis=-1)

    # ---- quadrature / pseudo-spectral projection --------------------------
    def quadrature(self, q):
        """Tensor Gauss-Hermite rule with q points per dimension (probabilists'), or, for q = "s<k>",
        the Smolyak sparse rule of level k (exact for total degree 2k+1), see sparse_quadrature."""
        if isinstance(q, str):
            return sparse_quadrature(self.d, int(q.lstrip("s")))
        x1, w1 = hermegauss(q)
        w1 = w1 / w1.sum()
        X = np.array(list(itertools.product(x1, repeat=self.d)))
        W = np.prod(np.array(list(itertools.product(w1, repeat=self.d))), axis=1)
        return X, W

    def psp_operator(self, q):
        """Returns (Psi_q (Q,P), Wpsi (Q,P)) so that coeffs = f(nodes) @ Wpsi."""
        X, W = self.quadrature(q)
        Psi = self.eval_basis(X)
        return Psi, Psi * W[:, None]

    def project(self, func, q):
        """Pseudo-spectral projection of a function of xi: func(X (Q,d)) -> (..., Q)."""
        X, W = self.quadrature(q)
        Psi = self.eval_basis(X)
        return np.asarray(func(X)) @ (Psi * W[:, None])


def sparse_quadrature(d, k):
    """Smolyak sparse Gauss-Hermite rule A(d+k, d) built from the i-point Gauss rules (exact for degree
    2i-1): sum over d <= |i| <= d+k, |i| >= k+1, of (-1)^(d+k-|i|) C(d-1, d+k-|i|) (x)_j Q_{i_j}.
    It is exact for polynomials of total degree 2k+1, so that the PC basis of degree p <= k is discretely
    orthonormal on it.  Coincident nodes are merged; some weights are negative."""
    rules = {i: hermegauss(i) for i in range(1, k + 2)}
    rules = {i: (x, w / w.sum()) for i, (x, w) in rules.items()}
    acc = {}
    for extra in multi_indices(d, k):                 # i = 1 + extra, |extra| <= k
        s = int(np.sum(extra))
        if s < k - d + 1:
            continue
        c = (-1) ** (k - s) * math.comb(d - 1, k - s)
        xs = [rules[e + 1][0] for e in extra]
        ws = [rules[e + 1][1] for e in extra]
        for pts, wts in zip(itertools.product(*xs), itertools.product(*ws)):
            key = tuple(np.round(pts, 12))
            acc[key] = acc.get(key, 0.0) + c * float(np.prod(wts))
    X = np.array([kk for kk, w in acc.items() if abs(w) > 1e-15])
    W = np.array([w for w in acc.values() if abs(w) > 1e-15])
    return X, W


def lognormal_coeffs(basis, mean_log, sigma, dim):
    """Exact PC coefficients of exp(mean_log + sigma*xi_dim)."""
    c = np.zeros(basis.P)
    A = basis.alphas
    pure = np.all(np.delete(A, dim, axis=1) == 0, axis=1)
    for k in np.where(pure)[0]:
        n = A[k, dim]
        c[k] = math.exp(mean_log + 0.5 * sigma ** 2) * sigma ** n / math.sqrt(math.factorial(n))
    return c


def linear_coeffs(basis, mean, sigma, dim):
    """Coefficients of mean + sigma*xi_dim."""
    c = np.zeros(basis.P)
    c[0] = mean
    c[basis.lin_index[dim]] = sigma
    return c

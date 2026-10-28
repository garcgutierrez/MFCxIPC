"""
Minimal NumPy multilayer perceptron with
  * forward-mode tangent propagation (directional derivative w.r.t. the input),
  * reverse-mode gradients of a loss that depends on both the output and its tangent,
  * Adam optimiser and flat-parameter interface for L-BFGS (scipy).

Row-vector convention: a_{l+1} = tanh(a_l W_l + b_l), output layer linear.
"""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0, act="tanh"):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            lim = np.sqrt(6.0 / (n_in + n_out))          # Glorot uniform
            self.W.append(rng.uniform(-lim, lim, (n_in, n_out)))
            self.b.append(np.zeros(n_out))
        self.act = act

    # ------------------------------------------------------------------
    @property
    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def get_flat(self):
        return np.concatenate([np.r_[W.ravel(), b] for W, b in zip(self.W, self.b)])

    def set_flat(self, theta):
        i = 0
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            self.W[l] = theta[i:i + W.size].reshape(W.shape); i += W.size
            self.b[l] = theta[i:i + b.size].copy(); i += b.size

    @staticmethod
    def flat_grads(gW, gb):
        return np.concatenate([np.r_[a.ravel(), c] for a, c in zip(gW, gb)])

    # ------------------------------------------------------------------
    def forward(self, x, xdot=None):
        """Returns y, ydot (None if xdot is None) and a cache for backward."""
        a, ad = x, xdot
        cache = []
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            zd = None if ad is None else ad @ self.W[l]
            cache.append((a, ad))
            if l < L - 1:
                a = np.tanh(z)
                s = 1.0 - a * a
                ad = None if zd is None else s * zd
                cache[-1] = cache[-1] + (a, s, zd)
            else:
                a, ad = z, zd
        return a, ad, cache

    def backward(self, cache, gy, gyd=None):
        """Gradients of a loss with dL/dy = gy and dL/dydot = gyd."""
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        gz, gzd = gy, gyd
        for l in range(L - 1, -1, -1):
            a_in, ad_in = cache[l][0], cache[l][1]
            gW[l] = a_in.T @ gz
            if gzd is not None:
                gW[l] += ad_in.T @ gzd
            gb[l] = gz.sum(axis=0)
            if l == 0:
                break
            ga = gz @ self.W[l].T
            gad = None if gzd is None else gzd @ self.W[l].T
            # through previous hidden activation: a = tanh(z), ad = s*zd
            _, _, a, s, zd = cache[l - 1]
            gz_prev = ga * s
            if gad is not None:
                gzd_prev = gad * s
                gz_prev = gz_prev + gad * zd * (-2.0 * a * s)
            else:
                gzd_prev = None
            gz, gzd = gz_prev, gzd_prev
        return gW, gb


class Adam:
    def __init__(self, n, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(n); self.v = np.zeros(n)
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    def step(self, theta, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * mh / (np.sqrt(vh) + self.eps)


class FourierMLP:
    """MLP on random Fourier features of the input (Tancik et al. 2020; Wang et al. 2021):
        gamma(x) = [x, sin(2 pi x B), cos(2 pi x B)],  B ~ N(0, sigma^2 / n_in) of shape (n_in, m), fixed.
    With inputs normalised so that |x|^2 / n_in = O(1), the phases 2 pi x B have standard deviation ~ 2 pi sigma.
    Same interface as MLP (forward with input tangent, backward, flat parameters); `sizes` are the outer sizes."""

    def __init__(self, sizes, seed=0, m=64, sigma=1.0, fseed=None):
        self.sizes = list(sizes)
        self.m, self.sigma = int(m), float(sigma)
        self.fseed = int(seed + 7919 if fseed is None else fseed)
        n_in = sizes[0]
        rng = np.random.default_rng(self.fseed)
        self.B2pi = 2 * np.pi * self.sigma * rng.normal(size=(n_in, self.m)) / np.sqrt(n_in)
        self.inner = MLP([n_in + 2 * self.m] + list(sizes[1:]), seed=seed)

    @property
    def W(self):
        return self.inner.W

    @property
    def n_params(self):
        return self.inner.n_params

    def get_flat(self):
        return self.inner.get_flat()

    def set_flat(self, theta):
        self.inner.set_flat(theta)

    def spec(self):
        return np.array([self.m, self.sigma, self.fseed], float)

    def forward(self, x, xdot=None):
        u = x @ self.B2pi
        su, cu = np.sin(u), np.cos(u)
        e = np.concatenate([x, su, cu], 1)
        ed = None
        if xdot is not None:
            ud = xdot @ self.B2pi
            ed = np.concatenate([xdot, cu * ud, -su * ud], 1)
        return self.inner.forward(e, ed)

    def backward(self, cache, gy, gyd=None):
        return self.inner.backward(cache, gy, gyd)       # the embedding has no trainable parameters


def make_net(sizes, seed=0, fourier=None):
    """fourier: None (plain MLP) or (m, sigma[, fseed]) for a FourierMLP."""
    if fourier is None:
        return MLP(sizes, seed=seed)
    f = list(fourier)
    return FourierMLP(sizes, seed=seed, m=int(f[0]), sigma=float(f[1]), fseed=None if len(f) < 3 else int(f[2]))


def net_spec(net):
    """Array stored with a model to rebuild its embedding (empty for a plain MLP)."""
    return net.spec() if isinstance(net, FourierMLP) else np.zeros(0)


def spec_from_file(z):
    s = z["fourier"] if "fourier" in z.files else np.zeros(0)
    return None if s.size == 0 else (int(s[0]), float(s[1]), int(s[2]))


def rad_indices(eps, n, rng, k=1.0, c=1.0):
    """Residual-based adaptive distribution (RAD, Wu et al. 2023): draw n of the candidates with probability
    proportional to eps^k / mean(eps^k) + c, where eps is the residual norm of each candidate."""
    e = np.asarray(eps, float) ** k
    p = e / max(e.mean(), 1e-300) + c
    p /= p.sum()
    return rng.choice(len(p), size=n, replace=True, p=p)

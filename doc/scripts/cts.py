from scipy.linalg import solve, lstsq, qr
from scipy.special import erf as erf
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

n = 20
dt, T = 1., int(1e4)

l = np.int(np.ceil(T / dt))
sqrt_dt = np.sqrt(dt)
sqrt_2 = np.sqrt(2)
rat = sqrt_dt / sqrt_2

w = np.random.uniform(-0.5, 0.5, size=(n, n))
w[np.diag_indices_from(w)] -= 2.0
w /= np.sqrt(n)

x = np.zeros((n, l))
x[:, 0] = np.random.uniform(-1, 1, size=n)
noise = np.random.normal(size=(n, l - 1))
for t in range(1, l):
    x[:, t] = x[:, t - 1] + w.dot(x[:, t - 1]) * dt + noise[:, t - 1] * sqrt_dt

plt.figure(figsize=(16, 4))
plt.plot(x[:, -100:].T)
plt.show()

x1 = x[:, :-1]
s = np.sign(np.diff(x))
c = (x - x.mean(1)[:, np.newaxis]).T
c1 = c[:-1].T

# cov_x = np.cov(x)
# mean_x = x.mean(1)
# x1_mean0 = x1 - mean_x[:, np.newaxis]

xq, xr = qr(x.T, mode='economic')


def back_sub(r, b):
    ans = np.empty(b.shape)
    for i in range(n - 1, -1, -1):
        ans[i] = b[i]
        for j in range(i + 1, n):
            ans[i] -= r[i, j] * ans[j]
        ans[i] /= r[i, i]
    return ans


def fit(i, iters=100):

    wi = np.ones(n) / float(n)  #* np.random.choice([-1, 1], size=n)

    # erf_last = erf(x1[i] * rat) + 1
    # erf_last = erf(x1[i]) + 1
    erf_last = np.inf

    e = []

    for it in range(iters):

        h = wi.dot(x1)

        # erf_next = erf(h * rat)
        erf_next = erf(h)
        ei = np.linalg.norm(erf_next - erf_last)
        e.append(ei)
        if ei * ei < 1e-5:
            break
        erf_last = erf_next.copy()

        h *= s[i] / erf_next

        # wi = solve(cov_x, x1_mean0.dot(h) / (l - 1))
        # wi = lstsq(x, x1_mean0.dot(h))[0]
        b = c1.dot(h)

        if False:
            wi = lstsq(x, b)[0]
        else:
            wi = xq.dot(back_sub(xr, b))

        wi = lstsq(c, wi)[0]

    print i, it, ei
    return wi, e[1:]


# pool = mp.Pool(processes=mp.cpu_count())
# res = pool.map(fit, range(n))
# pool.close()
# pool.terminate()
# pool.join()

res = [fit(i) for i in range(n)]

w_fit = np.empty((n, n))
w_fit = np.hstack([r[0] for r in res]) / rat
e = [r[1] for r in res]

w_flat = w.flatten()
w_fit_flat = w_fit.flatten()
plt.scatter(w_flat, w_fit_flat, c='k', s=0.1)
grid = np.linspace(w_flat.min(), w_flat.max())
plt.plot(grid, grid, 'r--', lw=0.5)
plt.show()

for ei in e:
    plt.plot(ei)
plt.show()

# h = np.random.uniform(size=c1.shape[1])
# b = c1.dot(h)

# # xw=b
# # r.dot(w) = q[:, p].T.dot(b)

# q, r, p = qr(x, pivoting=True, mode='economic')
# print np.allclose(x[:, p], q.dot(r))
# print np.allclose(x, q.dot(r[:, p]))

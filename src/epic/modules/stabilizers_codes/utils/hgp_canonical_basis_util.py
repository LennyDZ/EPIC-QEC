import numpy as np


def zero_till_i_vector(i, n):
    """Returns a vector of length n with zeros until index i and ones afterwards."""
    return np.concatenate((np.zeros(i, dtype=int), np.ones(n - i, dtype=int)))


def find_stl_basis(m):
    H = m.copy()
    m, n = H.shape
    K = np.eye(n, dtype=int)
    n_first_int = set(np.arange(n))
    idx_left = n_first_int.copy()
    for j in range(n):
        i = 0
        while i < m and H[i, j] == 0:
            i += 1
        if i == m:
            continue
        idx_left.remove(j)
        mask = zero_till_i_vector(j + 1, n) & (H[i, :])
        add = (
            np.kron(
                mask.reshape(1, n),
                np.array(H[:, j])[:, np.newaxis],
            )
            % 2
        )
        H = (H + add) % 2
        K_add = np.dot(
            K[:, j].reshape(n, 1),
            mask.reshape(1, n),
        )
        K += K_add
        K = K % 2
    # Extract relevant columns in a deterministic order.
    idx_left_sorted = sorted(idx_left)
    K = K[:, idx_left_sorted]
    F = np.eye(n, dtype=int)[:, idx_left_sorted]
    return K, F


def HGP(H1, H2, tilted=False):
    H1 = H1 % 2
    H2 = H2 % 2

    r1, n1 = H1.shape
    r2, n2 = H2.shape

    I_n1 = np.eye(n1, dtype=int)
    I_n2 = np.eye(n2, dtype=int)
    I_r1 = np.eye(r1, dtype=int)
    I_r2 = np.eye(r2, dtype=int)

    # Build Hx blocks
    Hx_left = np.kron(H1, I_n2)
    Hx_right = np.kron(I_r1, H2.T)
    Hz_left = np.kron(I_n1, H2)
    Hz_right = np.kron(H1.T, I_r2)
    if not tilted:
        Hx = np.hstack([Hx_left, Hx_right]) % 2
        Hz = np.hstack([Hz_left, Hz_right]) % 2
    else:
        Hx = np.hstack([Hx_left]) % 2
        Hz = np.hstack([Hz_left]) % 2
    return Hx, Hz


def get_canonical_basis(h1, h2, is_tilted=False):
    h1t, h2t = h1.copy().T, h2.copy().T
    Hx, Hz = HGP(h1, h2, tilted=is_tilted)
    n = Hx.shape[1]
    num_qubits = Hx.shape[1]

    K1, F1 = find_stl_basis(h1)
    K2, F2 = find_stl_basis(h2)
    K1t, F1t = find_stl_basis(h1t)
    K2t, F2t = find_stl_basis(h2t)

    logical_qubits = {}

    for i in range(K1.shape[1]):
        for h in range(K2.shape[1]):
            key = (i, h, "L")
            logical_qubits[key] = (
                np.kron(F1[:, i], K2[:, h]),
                np.kron(K1[:, i], F2[:, h]),
            )

    if not is_tilted:
        for j in range(K1t.shape[1]):
            for l in range(K2t.shape[1]):
                key = (j, l, "R")
                logical_qubits[key] = (
                    np.kron(K1t[:, j], F2t[:, l]),
                    np.kron(F1t[:, j], K2t[:, l]),
                )

    for key, val in logical_qubits.items():
        if key[2] == "L":
            r_size = n - len(val[0])
            logical_qubits[key] = (
                np.hstack([val[0], np.zeros(r_size)]),
                np.hstack([val[1], np.zeros(r_size)]),
            )
        elif key[2] == "R":
            l_size = n - len(val[0])
            logical_qubits[key] = (
                np.hstack([np.zeros(l_size), val[0]]),
                np.hstack([np.zeros(l_size), val[1]]),
            )

    return logical_qubits

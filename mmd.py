#!/usr/bin/env python
# encoding: utf-8

import torch

min_var_est = 1e-8


def _as_multiplier_list(kernel_multipliers):
    if isinstance(kernel_multipliers, str):
        kernel_multipliers = [
            value.strip()
            for value in kernel_multipliers.split(",")
            if value.strip()
        ]
    elif isinstance(kernel_multipliers, (int, float)):
        kernel_multipliers = [kernel_multipliers]

    multipliers = [float(value) for value in kernel_multipliers]
    if not multipliers:
        raise ValueError("kernel_multipliers must contain at least one value")
    if any(value <= 0 for value in multipliers):
        raise ValueError("all kernel_multipliers must be positive")
    return multipliers


def _off_diagonal_values(mat):
    n = mat.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    return mat[mask]


def _pairwise_squared_l2(X, Y):
    Z = torch.cat((X, Y), dim=0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    dist = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()
    return torch.clamp(dist, min=0.0)


def _print_distance_stats(dist_mat, median_sq_l2, multipliers, denominators):
    with torch.no_grad():
        vals = _off_diagonal_values(dist_mat)

        print("[L2 squared pairwise distance statistics]")
        print(f"mean   : {vals.mean().item():.6f}")
        print(f"median : {median_sq_l2.item():.6f}")
        print(f"25%    : {vals.quantile(0.25).item():.6f}")
        print(f"75%    : {vals.quantile(0.75).item():.6f}")
        print(f"min    : {vals.min().item():.6f}")
        print(f"max    : {vals.max().item():.6f}")
        print("kernel denominator multipliers:", multipliers)
        print("kernel denominators:", [value.item() for value in denominators])


def _mix_rbf_kernel(
    X,
    Y,
    kernel_multipliers,
    print_stats=True,
    eps=1e-12,
):
    """
    Gaussian kernel on squared L2 distance with median denominators.

    For each multiplier a:
        k_a(x, y) = exp(-||x-y||_2^2 / (a * median_sq_l2))

    median_sq_l2 is computed from the off-diagonal pairwise squared L2
    distances of concat(X, Y), then detached from the gradient graph.
    Passing several multipliers sums the corresponding Gaussian kernels.
    """

    assert X.size(0) == Y.size(0), "X and Y must have same batch size"
    m = X.size(0)
    multipliers = _as_multiplier_list(kernel_multipliers)

    D = _pairwise_squared_l2(X, Y)
    median_sq_l2 = torch.median(_off_diagonal_values(D)).detach()
    median_sq_l2 = torch.clamp(median_sq_l2, min=eps)

    denominators = [
        torch.clamp(median_sq_l2 * multiplier, min=eps)
        for multiplier in multipliers
    ]

    if print_stats:
        _print_distance_stats(D, median_sq_l2, multipliers, denominators)

    K = torch.zeros_like(D)
    for denominator in denominators:
        K += torch.exp(-D / denominator)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(multipliers)


def debug_kernel(K_XX, K_XY, K_YY):
    print("====== Kernel Stats ======")
    print("K_XX mean/std:", K_XX.mean().item(), K_XX.std().item())
    print("K_XY mean/std:", K_XY.mean().item(), K_XY.std().item())
    print("K_YY mean/std:", K_YY.mean().item(), K_YY.std().item())

    print("gap (XX - XY):", (K_XX.mean() - K_XY.mean()).item())

    print("====== Global ======")
    K = torch.cat([K_XX, K_XY, K_YY], dim=0)
    print("K mean/std:", K.mean().item(), K.std().item())
    print("K min/max:", K.min().item(), K.max().item())


def mix_rbf_mmd2(X, Y, kernel_multipliers, biased=True, print_stats=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(
        X,
        Y,
        kernel_multipliers,
        print_stats=print_stats,
    )
    if print_stats:
        debug_kernel(K_XX, K_XY, K_YY)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, kernel_multipliers, biased=True, print_stats=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(
        X,
        Y,
        kernel_multipliers,
        print_stats=print_stats,
    )
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return  mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)                     # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est

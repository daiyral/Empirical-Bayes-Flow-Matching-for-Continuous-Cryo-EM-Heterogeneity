import torch


def _conjugate_gradient_batched(A, sigma_y2, c_t, rhs, max_iters=5, tol=1e-6):
    # solve (sigma_y^2 I + c_t A A^T) v = rhs per sample; A: (n, M, D), rhs: (n, M)
    def matvec(v):
        AtV = torch.bmm(A.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)
        AAtV = torch.bmm(A, AtV.unsqueeze(-1)).squeeze(-1)
        return sigma_y2 * v + c_t * AAtV

    v = torch.zeros_like(rhs)
    r = rhs - matvec(v)
    p = r.clone()
    rs_old = (r * r).sum(dim=-1, keepdim=True)
    for _ in range(max_iters):
        Ap = matvec(p)
        denom = (p * Ap).sum(dim=-1, keepdim=True).clamp_min(1e-30)
        alpha = rs_old / denom
        v = v + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=-1, keepdim=True)
        if rs_new.max() < tol:
            break
        beta = rs_new / rs_old.clamp_min(1e-30)
        p = r + beta * p
        rs_old = rs_new
    return v


def _direct_solve_batched(A, sigma_y2, c_t, rhs):
    n, M, _ = A.shape
    AAt = torch.bmm(A, A.transpose(1, 2))
    Imat = torch.eye(M, device=A.device, dtype=A.dtype).expand(n, M, M)
    K = sigma_y2 * Imat + c_t * AAt
    return torch.linalg.solve(K, rhs.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def posterior_ddpm_sample(net, A, y, schedule, sigma_y, N=50, cg_iters=5,
                          use_direct_solve_below_M=16, forward_kind="linear",
                          c_t_max=10.0, correction_max_norm=3.0):
    # DiEM-iso: ancestral DDPM where each denoiser output is corrected with the
    # isotropic-covariance posterior mean, x_hat + c_t J^T (sigma_y^2 I + c_t J J^T)^-1 (y - F(x_hat))
    net.eval()
    device = A.device
    schedule.to(device)
    n, M, D = A.shape
    T = schedule.T

    sigma_y2 = float(sigma_y) ** 2

    stride = T // N
    timesteps = list(range(T - 1, -1, -stride))
    if timesteps[-1] != 0:
        timesteps.append(0)

    x = torch.randn(n, D, device=device)
    abar = schedule.abar
    eps_floor = 1e-5

    use_direct = M < use_direct_solve_below_M

    for k, t in enumerate(timesteps):
        t_norm = torch.full((n,), (t + 1) / T, device=device)
        x_hat_1 = net(x, t_norm)

        abar_t = abar[t].clamp(min=eps_floor).item()
        c_t = (1.0 - abar_t) / max(abar_t, eps_floor)
        c_t = min(c_t, c_t_max)

        Ax = torch.bmm(A, x_hat_1.unsqueeze(-1)).squeeze(-1)
        if forward_kind == "linear":
            J = A
            F_x = Ax
        elif forward_kind == "sin":
            # Gauss-Newton linearization: J = diag(cos(A x_hat)) A
            J = A * torch.cos(Ax).unsqueeze(-1)
            F_x = torch.sin(Ax)
        else:
            raise ValueError(f"unknown forward_kind: {forward_kind!r}")
        residual = y - F_x

        if use_direct:
            v = _direct_solve_batched(J, sigma_y2, c_t, residual)
        else:
            v = _conjugate_gradient_batched(J, sigma_y2, c_t, residual, max_iters=cg_iters)
        delta = c_t * torch.bmm(J.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)

        if correction_max_norm is not None:
            norm = delta.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(eps_floor)
            delta = delta * (correction_max_norm / norm).clamp(max=1.0)
        x_hat_corrected = x_hat_1 + delta

        if k == len(timesteps) - 1:
            x = x_hat_corrected
            break

        abar_t_t = abar[t].clamp(min=eps_floor)
        one_minus_t = (1.0 - abar_t_t).clamp(min=eps_floor)
        t_prev = timesteps[k + 1]
        abar_prev = abar[t_prev].clamp(min=eps_floor)
        one_minus_prev = (1.0 - abar_prev).clamp(min=eps_floor)
        alpha_bar_step = abar_t_t / abar_prev
        one_minus_step = (1.0 - alpha_bar_step).clamp(min=0.0)

        coef_x_hat = abar_prev.sqrt() * one_minus_step / one_minus_t
        coef_xt = alpha_bar_step.sqrt() * one_minus_prev / one_minus_t
        mean = coef_x_hat * x_hat_corrected + coef_xt * x

        var = one_minus_step * one_minus_prev / one_minus_t
        std = var.clamp(min=0.0).sqrt()
        x = mean + std * torch.randn_like(x)

    return x.detach()

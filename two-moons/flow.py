import torch

from data import forward_op


def flow_loss(net, x1, t_eps=1e-3):
    # v-loss with an x-prediction net: || (x_hat_1 - x_1) / (1 - t) ||^2
    B = x1.shape[0]
    t = torch.rand(B, device=x1.device) * (1.0 - 2.0 * t_eps) + t_eps
    x0 = torch.randn_like(x1)
    t_b = t.view(B, *([1] * (x1.dim() - 1)))
    xt = (1.0 - t_b) * x0 + t_b * x1
    x_hat_1 = net(xt, t)
    weight = 1.0 / (1.0 - t_b)
    return ((x_hat_1 - x1) * weight).pow(2).sum(dim=-1).mean()


def guided_flow_sample(net, A, y, eta, N, forward_kind="linear", t_eps_sample=1e-3):
    # Euler step and rendering-gradient correction applied to x_t in one update:
    #   x <- x + dt * v - eta * grad_{x_hat_1} ||F(A x_hat_1) - y||^2
    net.eval()
    n, _, D = A.shape
    x = torch.randn(n, D, device=A.device)
    ts = torch.linspace(0.0, 1.0 - t_eps_sample, N + 1, device=A.device)

    for k in range(N):
        t = ts[k]
        dt = (ts[k + 1] - t).item()
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            x_hat_1 = net(x_in, t.expand(n))
            diff = forward_op(torch.einsum("nmd,nd->nm", A, x_hat_1), forward_kind) - y
            grad = torch.autograd.grad(diff.pow(2).sum(), x_hat_1, create_graph=False)[0]
        v = (x_hat_1.detach() - x) / (1.0 - t.item())
        x = x + dt * v - eta * grad.detach()

    with torch.no_grad():
        x_hat_1 = net(x, ts[-1].expand(n))
    return x_hat_1.detach()

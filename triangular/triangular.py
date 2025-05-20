import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


# class MonotonicBlock(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.a = nn.Parameter(torch.rand(1))         # coefficient for x_k
#         self.b = nn.Parameter(torch.rand(1))         # coefficient for erf(x_k)
#         self.c = nn.Parameter(torch.zeros(1))        # bias term


#     def forward(self, x):
#         # Apply the monotonic transformation
#         x = torch.exp(self.a) * x  + self.c
#         return x    

class MonotonicBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, n_integration_points=50):
        super().__init__()
        self.n_integration_points = n_integration_points
        self.net = NonMonotonicBlock(input_dim=input_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        x: [B, input_dim] where the last dimension is x_k (the one we integrate over)
        """
        x_prev = x[:, :-1]  # x1, ..., x_{k-1}
        xk = x[:, -1:]      # x_k, shape [B, 1]
        B = x.size(0)

        # Integration range: from 0 to x_k
        xk_vals = torch.linspace(0, 1, self.n_integration_points, device=x.device).view(1, -1)  # [1, N]
        xk_vals = xk_vals * xk  # [B, N]

        # Repeat x_prev to match integration points
        x_prev_rep = x_prev.unsqueeze(1).expand(-1, self.n_integration_points, -1)  # [B, N, k-1]
        xk_rep = xk_vals.unsqueeze(-1)  # [B, N, 1]

        # Combine inputs
        x_full = torch.cat([x_prev_rep, xk_rep], dim=-1).reshape(-1, x.size(1))  # [B*N, k]

        # Evaluate ĝ and rectify
        g_hat = self.net(x_full)  # [B*N, 1]
        g_hat = self.softplus(g_hat)  # ensure positivity
        g_hat = g_hat.view(B, self.n_integration_points)  # [B, N]

        # Approximate integral over x_k using the trapezoidal rule
        dx = xk / (self.n_integration_points - 1)
        integral = torch.sum(g_hat, dim=1) * dx.squeeze(1)  # [B]

        return integral.unsqueeze(1)  # [B, 1]



class NonMonotonicBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))

        self.g = nn.Sequential(*layers)
        
    def forward(self, x):
        # Apply the non-monotonic transformation
        x = self.g(x)
        return x

class TriangularBlock(nn.Module):
    def __init__(self,input_dim, hidden_dim, hidden_layers):
        super().__init__()
        self.monotonic = MonotonicBlock(input_dim, hidden_dim, hidden_layers)
        self.input_dim = input_dim
        self.non_monotonic = NonMonotonicBlock(max(1,input_dim-1), hidden_dim, hidden_layers)
    
    def forward(self, x):
        #print("x", x.shape)
        if self.input_dim == 1:
            g = 0.0
            xk = x
        else:
            x_prev = x[:, :-1]
            xk = x[:, -1].unsqueeze(1)  # shape: [batch, 1]
            #print("x_prev", x_prev.shape)
            g = self.non_monotonic(x_prev)
        # Apply the monotonic and non-monotonic transformations
        f = self.monotonic(x)
        return g + f



class TriangularTransport(nn.Module):
    def __init__(self, config:SimpleNamespace):
        super().__init__()    
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.hidden_layers = config.hidden_layers
        self.maps = nn.ModuleList()
        for i in range(self.input_dim):
            self.maps.append(TriangularBlock(i+1, self.hidden_dim, self.hidden_layers))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


    def forward(self, x):
        # Apply the triangular transformation
        z = torch.zeros_like(x)
        # print("x", x.shape)
        # print("z", z.shape)
        for i in range(self.input_dim):
            z_i = self.maps[i](x[:, :i+1]).squeeze(1)  # shape: [batch, 1]
            #print("z_i", z_i.shape)
            z[:, i] = z_i
        return z
    
    def inverse(self, z, max_iter=50, tol=1e-6):
        x = torch.zeros_like(z)

        for i in range(self.input_dim):
            def forward_at_xi(xi_val, prev_x):
                x_full = torch.cat([prev_x, xi_val], dim=1)  # both 2D now
                return self.maps[i](x_full).squeeze(1)

            xi = z[:, i].clone()
            prev_x = x[:, :i]

            for _ in range(max_iter):
                xi_val = xi.clone().detach().requires_grad_(True).unsqueeze(1)
                zi_pred = forward_at_xi(xi_val, prev_x)
                residual = zi_pred - z[:, i]

                if torch.all(residual.abs() < tol):
                    break

                grad = torch.autograd.grad(zi_pred.sum(), xi_val)[0].squeeze(1)
                grad = grad.clamp(min=1e-4)  # prevent explosion
                delta = (residual / grad).clamp(-3.0, 3.0)  # clip update
                xi = xi - delta

            x[:, i] = xi

        return x
    
    def sample(self, num_samples):
        """
        Sample from the normalizing flow.
        """
        z = torch.randn(num_samples, self.input_dim).to(self.config.device)
        x = self.inverse(z)
        return x, z
    
def triangular_transport_loss(z, x):
    """
    Compute triangular transport loss from predicted z = S(x), and input x.

    Args:
        z: [B, D] - output of the triangular map
        x: [B, D] - input to the triangular map

    Returns:
        scalar loss
    """
    log_diag_jacobian = []
    for k in range(z.shape[1]):
        zk = z[:, k]
        grad_k = torch.autograd.grad(
            zk.sum(), x, create_graph=True, retain_graph=True
        )[0][:, k]  # ∂zk / ∂xk
        log_diag_jacobian.append(torch.log(torch.abs(grad_k) + 1e-8))

    log_det = torch.stack(log_diag_jacobian, dim=1).sum(dim=1)  # sum log ∂Sk/∂xk
    norm_term = 0.5*(z ** 2).sum(dim=1)

    return (norm_term - log_det).mean()


def separable_triangular_loss(model, x, l1_weight=0.0001):
    """
    Compute the separable loss for triangular transport maps.
    
    Args:
        model: TriangularTransport instance
        x: input tensor [B, D] with requires_grad=True
    Returns:
        scalar loss
    """
    B, D = x.shape
    total_loss = 0.0

    for k in range(D):
        # Input to S_k is x[:, :k+1]
        x_sub = x[:, :k+1].clone().detach().requires_grad_(True)

        # Forward through k-th triangular block
        S_k_x = model.maps[k](x_sub).squeeze()  # [B]
        term1 = 0.5 * S_k_x.pow(2)

        # Compute ∂Sk/∂xk (i.e. last input to this block)
        grad_k = torch.autograd.grad(
            S_k_x.sum(), x_sub, create_graph=True, retain_graph=True
        )[0][:, -1]  # [B], only ∂/∂x_k

        term2 = torch.log(torch.abs(grad_k) + 1e-8)
        loss_k = term1 - term2

        total_loss += loss_k.mean()
    if l1_weight > 0:
        l1_loss = sum(p.abs().sum() for p in model.parameters())
        total_loss += l1_weight * l1_loss

    return total_loss




if __name__ == "__main__":
    model = TriangularTransport(SimpleNamespace(input_dim=2, hidden_dim=32, hidden_layers=2))
    x = torch.randn(5, 2)  # Example input
    z = model(x)              # Forward
    x_inv = model.inverse(z)  # Inverse

    print(torch.max(torch.abs(x - x_inv)))  # should be close to 0
    print("x", x)
    print("z", z)
    print("x_inv", x_inv)
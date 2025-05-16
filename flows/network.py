import torch 
import torch.nn as nn


class AffineCoupling(nn.Module):
    def __init__(self, config):
        """
        Initialize the flow network.
        """
        super(AffineCoupling, self).__init__()
        self.config = config
        layers = []
        layers.append(nn.Linear(config.input_dim//2, config.hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.layers = nn.Sequential(*layers)



    def forward(self, x1):
        params = self.layers(x1)
        translate, log_scale = params.chunk(2, dim=1)
        log_scale = torch.clamp(log_scale, min=-5.0, max=5.0)  
        scale = torch.exp(log_scale)
        log_determinant = torch.sum(log_scale, dim=1)
        return translate, scale, log_determinant
    
    def inverse(self, z1):
        params = self.layers(z1)
        translate, log_scale = params.chunk(2, dim=1)
        log_scale = torch.clamp(log_scale, min=-5.0, max=5.0)  
        scale = torch.exp(log_scale)
        return translate, scale
    
    
    
class Permutation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        perm = torch.randperm(dim)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x):
        return x[:, self.perm]

    def inverse(self, x):
        return x[:, self.inv_perm]



def flow_loss(z, log_det_J):
    """
    z: latent vector from f(x)
    log_det_J: log-determinant of the Jacobian
    """
    # Assuming standard normal prior on z
    log_pz = -0.5 * torch.sum(z**2, dim=1)  # per sample
    loss = - (log_pz + log_det_J)           # negative log-likelihood
    return loss.mean()   


class NormalizingFlow(nn.Module):
    def __init__(self, config):
        """
        Initialize the normalization flow.
        """
        super(NormalizingFlow, self).__init__()
        self.config = config
        self.transformations = nn.ModuleList()
        self.permutations = nn.ModuleList()
        self.nflows = config.nflows
        for _ in range(self.nflows):
            self.transformations.append(AffineCoupling(config))
            self.permutations.append(Permutation(config.input_dim))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
    
    def forward(self, x):
        """
        Forward pass through the normalization flow.
        """
        sum_log_det = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.nflows):
            # Split the input into two parts
            x1, x2 = x.chunk(2, dim=1)
            #print("x1", x1.shape, "x2", x2.shape)
            # Apply the flow network to the second part
            translate, scale, log_det = self.transformations[i](x1)
            #print("translate", translate.shape, "scale", scale.shape, "log_det", log_det.shape)
            z2 = x2 * scale + translate
            sum_log_det += log_det
            # Concatenate the two parts
            x = torch.cat([x1, z2], dim=1)
            # permute the input
            x = self.permutations[i](x)
        return x, sum_log_det
    
    def inverse(self, z):
        for i in reversed(range(self.nflows)):
            # reverse permute the input
            z = self.permutations[i].inverse(z)
            z1, z2 = z.chunk(2, dim=1)
            # Apply the inverse of the flow network to the second part
            translate, scale = self.transformations[i].inverse(z1)
            x2 = (z2 - translate) / scale
            # Concatenate the two parts
            z = torch.cat([z1, x2], dim=1)
        return z
    
    def sample(self, num_samples):
        """
        Sample from the normalizing flow.
        """
        z = torch.randn(num_samples, self.config.input_dim).to(self.config.device)
        x = self.inverse(z)
        return x, z
            
            

        


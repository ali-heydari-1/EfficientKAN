import torch
import torch.nn.functional as F
import math


class KANLinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=3, activation_fn=torch.nn.SiLU, grid_limits=[-1, 1]):
        super(KANLinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        interval = (grid_limits[1] - grid_limits[0]) / grid_size
        self.grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * interval + grid_limits[0])
                     .expand(input_dim, -1).contiguous())

        self.base_weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim, grid_size + spline_order))
        self.spline_scaler = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))

        self.activation_fn = activation_fn()

        self.initialize_parameters()

    def initialize_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        torch.nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def b_spline_bases(self, x: torch.Tensor):
        grid = self.grid
        x = x.unsqueeze(-1)

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-k-1]) / (grid[:, k:-1] - grid[:, :-k-1]) * bases[:, :, :-1]
            ) + (
                (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:-k]) * bases[:, :, 1:]
            )

        return bases

    def compute_coefficients(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.input_dim
        assert y.size() == (x.size(0), self.input_dim, self.output_dim)

        A = self.b_spline_bases(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (self.output_dim, self.input_dim, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def weighted_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1))

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.input_dim)

        base_output = F.linear(self.activation_fn(x), self.base_weight)
        spline_output = F.linear(
            self.b_spline_bases(x).view(x.size(0), -1),
            self.weighted_spline_weight.view(self.output_dim, -1),
        )
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.output_dim)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        with torch.no_grad():
            assert x.dim() == 2 and x.size(1) == self.input_dim
            splines = self.b_spline_bases(x).permute(1, 0, 2)
            orig_coeff = self.weighted_spline_weight.permute(1, 2, 0)
            spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

            x_sorted = torch.sort(x, dim=0)[0]
            uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
            grid_uniform = (torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
                            * uniform_step
                            + x_sorted[0]
                            - margin)

            grid = torch.cat([
                grid_uniform[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid_uniform,
                grid_uniform[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ], dim=0)

            self.grid.data.copy_(grid.clone().T)
            self.spline_weight.data.copy_(self.compute_coefficients(x, spline_output))


class KANetwork(torch.nn.Module):
    def __init__(self, hidden_layers, grid_size=5, spline_order=3, activation_fn=torch.nn.SiLU, grid_limits=[-1, 1]):
        super(KANetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for input_dim, output_dim in zip(hidden_layers, hidden_layers[1:]):
            self.layers.append(KANLinearLayer(input_dim, output_dim, grid_size=grid_size, spline_order=spline_order, activation_fn=activation_fn, grid_limits=grid_limits))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            layer.update_grid(x)
            x = layer(x)
        return x

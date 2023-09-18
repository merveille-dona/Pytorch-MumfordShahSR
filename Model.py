import torch
import torch.nn
import torch.autograd

import numpy

import Utils

class CircularConv2d(torch.nn.Conv2d):

    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None
        ) -> None:
        
        self.kernel_size = kernel_size
        
        super(CircularConv2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding='valid', 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode=padding_mode, 
            device=device, 
            dtype=dtype
        )
        
    def T(self, x: torch.Tensor) -> torch.Tensor:
        x_flipped = torch.flip(x, dims=[0, 1])
        out = self.forward(x_flipped)
        return torch.flip(out, dims=[0, 1])
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #pad_width = [ (0, 0) ] + [  (i // 2, i // 2) for i in self.kernel_size ]
        n = self.kernel_size[0] // 2
        m = self.kernel_size[1] // 2
        # pad_width = [ (0, 0) ] + [ (i // 2, i // 2) for i in self.kernel_size ]
        pad_width = [ (0, 0), (n, n), (m, m) ]
        x_padded = numpy.pad(x.cpu().detach().numpy(), pad_width=pad_width , mode='wrap')
        x_padded = torch.tensor(x_padded, dtype= x.dtype, device=x.device)
        return super().forward(x_padded)


# class Kernel(torch.nn.Module):

#     def __init__(self, 
#         in_channels: int,
#         out_channels: int,
#         kernel_size: tuple
#     ) -> None:

#         super(Kernel, self).__init__()

#         self.conv = torch.nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=1,
#             padding='same',
#             dilation=1,
#             bias=False
#         )

#         # torch.nn.init.zeros_(self.conv.weight.data)
#         torch.nn.init.constant_(self.conv.weight.data, 1.0/(kernel_size[0]*kernel_size[0]))
#         # torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.0)

#         self.batchnorm = torch.nn.BatchNorm2d(
#             num_features=out_channels,
#             eps=1e-05,
#             momentum=1e-1,
#             affine=True
#         )

#         self.activation = torch.nn.Sigmoid()

#         # self.dropout = torch.nn.Dropout(p=0.2)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         out = self.conv(x)
#         # out = self.batchnorm(out.unsqueeze(0)).squeeze(0)
#         # out = self.activation(out)
#         # # out = self.dropout(out4)

        
#         return out

      

class Iteration(torch.nn.Module):

    # Static attribute
    # <=> if attributs change, all instance change
    # <=> attribute shared between all "Iteration" object 
    # https://docs.python.org/3/tutorial/classes.html#class-and-instance-variables
    # This attribute are not learnable
    #d_x: torch.Tensor # Shared
    #d_y: torch.Tensor # Shared
    #b_x: torch.Tensor # Shared
    #b_y: torch.Tensor # Shared
    # f: torch.Tensor # shared

    def __init__(self, 
        nb_intermediate_channels: int, 
        kernel_size: tuple,
        alpha: float,
        beta0: float,
        beta1: float,
        sigma: float,
        alpha_learnable: bool,
        beta0_learnable: bool,
        beta1_learnable: bool,
        sigma_learnable: bool,
        taylor_nb_iterations: int,
        taylor_kernel_size: tuple
    ) -> None:

        super(Iteration, self).__init__()
        # f_approx = argmin { 
        #   (alpha / 2) || g - Hf ||^{2}_{2}
        #   + (beta0 / 2) || nabla f ||^{2}_{2}
        #   + beta1 || nabla f ||_{1}

        self.nb_intermediate_channels = nb_intermediate_channels
        self.kernel_size = kernel_size
        self.n = taylor_nb_iterations
        
        # Hyper-parameters
        # self.alpha = torch.nn.Parameter(data=torch.abs(torch.randn(1, dtype=torch.float)), requires_grad=True)
        # self.beta1 = torch.nn.Parameter(data=torch.abs(torch.randn(1, dtype=torch.float)), requires_grad=True)
        self.alpha = torch.nn.Parameter(
            data=torch.tensor([alpha], dtype=torch.float),
            requires_grad=alpha_learnable
        )

        self.beta0 = torch.nn.Parameter(
            data=torch.tensor([beta0], dtype=torch.float),
            requires_grad=beta0_learnable
        )

        self.beta1 = torch.nn.Parameter(
            data=torch.tensor([beta1], dtype=torch.float),
            requires_grad=beta1_learnable
        )

        self.sigma = torch.nn.Parameter(
            data=torch.tensor([sigma], 
            dtype=torch.float), 
            requires_grad=sigma_learnable
        )

        # # H
        self.h = CircularConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=False
        )

        # torch.nn.init.xavier_uniform_(self.h.weight.data)
        # self.batchnorm = torch.nn.BatchNorm2d(
        #     num_features=out_channels,
        #     eps=1e-05,
        #     momentum=1e-1,
        #     affine=True
        # )

        # self.activation = torch.nn.Tanh()
        
        #self.activation = torch.nn.LeakyReLU(0.1)
        
        
        # self.relu = torch.nn.ReLU()
        # threshold = 1e-10
        # value = threshold
        # self.threshold = torch.nn.Threshold(threshold, value, inplace=False)
        # self.parameter_activation = torch.nn.Threshold(threshold, value, inplace=False)
        # self.parameter_activation = torch.nn.Sigmoid()
        # self.parameter_activation = lambda x : x
        # self.taylor_activation = torch.nn.Tanh()
        # self.taylor_activation = torch.nn.LeakyReLU(negative_slope=1e-1)
        # self.taylor_activation = torch.nn.Sigmoid()
        # self.taylor_activation = lambda u : u
        # torch.nn.Hardtanh(min_val, max_val, inplace=False)

        # self.ht_decim_activation = torch.nn.Sigmoid()
        # self.ht_decim_activation = lambda x : x


         
    def forward(self, STg, decim_row, decim_col, d_x, d_y, b_x, b_y) -> torch.Tensor:
        
        # STg = S^{T} g 
        #print('STg :', STg)
        #print(self.sigma.device)
        
        # COMPUTE f approximation
        gradT_x = Utils.dxT(d_x - b_x)
        # gradT_x_normalized = Utils.matrix_normalize(gradT_x)
        # print('gradT_x :', gradT_x, torch.min(gradT_x), torch.max(gradT_x))
        ## = (nabla_x)^{T} (d_x - b_x)
        gradT_y = Utils.dyT(d_y - b_y)
        # gradT_y_normalized = Utils.matrix_normalize(gradT_y)
        # print('gradT_y :', gradT_y, torch.min(gradT_y), torch.max(gradT_y))
        ## = (nabla_y)^{T} (d_y - b_y)
        sigma_expr = self.sigma * ( gradT_x + gradT_y )
        # sigma_expr = self.sigma * ( gradT_x_normalized + gradT_y_normalized )
        # sigma_expr = self.sigma_x * gradT_x_normalized + self.sigma_y * gradT_y_normalized
        # print('sigma_expr :', sigma_expr, torch.min(sigma_expr), torch.max(sigma_expr))
        ## = sigma * [ (nabla_x)^{T} (d_x - b_x) + (nabla_y)^{T} (d_y - b_y) ]
        alpha_expr = self.alpha * self.h.T(STg.unsqueeze(0))
        # print('alpha_expr :', alpha_expr, torch.min(alpha_expr), torch.max(alpha_expr))
        # alpha_expr = alpha_expr.squeeze(0)
        ## = alpha * (H^{T} S^{T} g)
        # Iteration.f = self.conv1((sigma_expr + alpha_expr).unsqueeze(0))
        # f = self.inv(sigma_expr + alpha_expr)
        # f = f.squeeze(0)
        # if gradT_x.isnan().any():
        #     raise AssertionError('gradT_x : Nan ')
        # if gradT_y.isnan().any():
        #     raise AssertionError('gradT_y : Nan ')
                
        # if (sigma_expr + alpha_expr).isnan().any():
        #     raise AssertionError('Nan before inversion')
        f = self.taylor_young_ld(
            x = (sigma_expr + alpha_expr).squeeze(0), 
            decim_row = decim_row,
            decim_col = decim_col,
            n = self.n
        )
        # print('f :', f, torch.min(f), torch.max(f))
        # if f.isnan().any():
        #     raise AssertionError('Nan after inversion')

        ## = [ alpha H^{T} S^{T} S H + (beta0 + sigma) laplacian ]^{-1}
        ##  * (
        ##      sigma * [ (nabla_x)^{T} (d_x - b_x) + (nabla_y)^{T} (d_y - b_y) ]
        ##      + alpha * (H^{T} S^{T} g)
        ##  )

        # Update (d_x, d_y) : Multidimensional Soft Thresholding
        dx_f = Utils.dx(f)
        # print('dx_f :', dx_f, torch.min(dx_f), torch.max(dx_f))
        dy_f = Utils.dy(f)
        # print('dy_f :', dy_f, torch.min(dy_f), torch.max(dy_f))

        # if dx_f.isnan().any():
        #     raise AssertionError('dx_f : Nan ')
        # if dy_f.isnan().any():
        #     raise AssertionError('dy_f : Nan ')
        
        # d_x = Utils.soft((dx_f + b_x).unsqueeze(0), self.beta1 / self.sigma_x)
        # d_y = Utils.soft((dy_f + b_y).unsqueeze(0), self.beta1 / self.sigma_y)

        d_x, d_y = Utils.multidimensional_soft(
            torch.concat(
                [ 
                    (dx_f + b_x).unsqueeze(0), 
                    (dy_f + b_y).unsqueeze(0)
                ],
                0
            ),
            self.beta1 / self.sigma
            # self.parameter_activation(self.beta1 / self.sigma)
        )
        # print('d_x :', d_x)
        # print('d_y :', d_y)


        # if d_x.isnan().any():
        #     raise AssertionError('d_x : Nan ')
        # if d_y.isnan().any():
        #     raise AssertionError('d_y : Nan ')
      
        # Update (b_x, b_y)
        b_x += (dx_f - d_x)
        b_y += (dy_f - d_y)

        # print('b_x :', b_x)
        # print('b_y :', b_y)

        # print()
        # print()
        # print()
        # print()
        # print()
        
        return [ f, d_x, d_y, b_x, b_y ]


    def taylor_young_ld(self, x: torch.Tensor, decim_row: int, decim_col: int, n: int) -> torch.Tensor:

        # k = 0
        ld = x
        for k in range(1, n+1):
            # ld = x - self.taylor_activation(self.compute(ld, decim_row, decim_col))
            ld = x - self.compute(ld, decim_row, decim_col)
            # print('x :', x)

        # return self.taylor_activation(ld)
        return ld

    def compute(self, u: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
        """Computes :
        [I - (alpha H^{T} S^{T} S H + (beta0 + sigma) laplacian)] u
        = u - [(alpha H^{T} S^{T} S H] u - [(beta0 + sigma) laplacian)] u
        """
        
        # [ alpha H^{T} S^{T} S H ] u
        out1 = self.h(u.unsqueeze(0))
        out2 = Utils.decimation(out1.squeeze(0), decim_row, decim_col)
        out3 = Utils.decimation_adjoint(out2, decim_row, decim_col)
        out4 = self.h.T(out3.unsqueeze(0))
        out5 = self.alpha * out4
        term1 = out5.squeeze(0)

        # [(beta0 + sigma) laplacian ] u
        laplacian = Utils.laplacian2D_v2(u)
        term2 = (self.beta0 + self.sigma) * laplacian
      
        # [ I - (alpha H^{T} S^{T} S H + (beta0 + sigma) laplacian) ] u
        #= u - [ (alpha H^{T} S^{T} S H ] u - [ (beta0 + sigma) laplacian) ] u
        res = u - term1 - term2

        return res


class Unfolding(torch.nn.Module):

    def __init__(self, 
        nb_intermediate_channels: int,
        kernel_size: tuple,
        nb_iterations: int,
        alpha: float,
        beta0: float,
        beta1: float,
        sigma: float,
        alpha_learnable: bool,
        beta0_learnable: bool,
        beta1_learnable: bool,
        sigma_learnable: bool,
        taylor_nb_iterations: int,
        taylor_kernel_size: tuple
    ) -> None:
        
        super(Unfolding, self).__init__()
        
        params = [
            nb_intermediate_channels,
            kernel_size,
            alpha,
            beta0,
            beta1,
            sigma,
            alpha_learnable,
            beta0_learnable,
            beta1_learnable,
            sigma_learnable,
            taylor_nb_iterations,
            taylor_kernel_size
        ]


        iters = [ Iteration(*params) for _ in range(0, nb_iterations) ]
        
        # self.iterations = torch.nn.Sequential(*iters)
        
        self.iterations = torch.nn.ModuleList(iters)


    def forward(self, 
        low_resolution: torch.Tensor,
        decim_row: int,
        decim_col: int
    ) -> torch.Tensor:

        """
        
            Params:
                - low_resolution : image low-resolution
                - decim_row : decimation on line
                - decim_col : decimation on col

            Return:
                Image high-resolution of size.
                If size of low_resolution is (N, M), Image high-resolution
                will be (N*decim_row, M*decim_col)

        """


        # Initialize static attribute / shared attribute

        g = low_resolution
        STg = Utils.decimation_adjoint(g, decim_row, decim_col)
        
        d_x = torch.zeros_like(STg)
        d_y = torch.zeros_like(STg)
        b_x = torch.zeros_like(STg)
        b_y = torch.zeros_like(STg)

        # _ = self.iterations(inputs)
        for iter_layer in self.iterations:
            # STg, d_x, d_y, b_x, b_y = iter_layer(STg, d_x, d_y, b_x, b_y)
            f, d_x, d_y, b_x, b_y = iter_layer(STg, decim_row, decim_col, d_x, d_y, b_x, b_y)

        # f_approx = Iteration.f
        f_approx = f
        # Normalize f
        mini = torch.min(f_approx)
        maxi = torch.max(f_approx)
        normalized = (f_approx - mini) / (maxi - mini)

        return normalized

    @classmethod
    def from_config(cls, config: dict) -> 'Unfolding':

        model_config = config['model']
        params = model_config['params']

        nb_intermediate_channels = params['nb_intermediate_channels']
        kernel_size = params['kernel_size']
        nb_iterations = params['nb_iteration']

        alpha = params['alpha']['initialize']
        beta0 = params['beta0']['initialize']
        beta1 = params['beta1']['initialize']
        sigma = params['sigma']['initialize']

        alpha_learnable = params['alpha']['is_learnable']
        beta0_learnable = params['beta0']['is_learnable']
        beta1_learnable = params['beta1']['is_learnable']
        sigma_learnable = params['sigma']['is_learnable']

        taylor_nb_iterations = params['taylor']['nb_iteration']
        taylor_kernel_size = params['taylor']['kernel_size']

        params = [
            nb_intermediate_channels,
            kernel_size,
            nb_iterations,
            alpha,
            beta0,
            beta1,
            sigma,
            alpha_learnable,
            beta0_learnable,
            beta1_learnable,
            sigma_learnable,
            taylor_nb_iterations,
            taylor_kernel_size
        ]

        model = Unfolding(*params)
        device = model_config['device']

        return model.to(device)
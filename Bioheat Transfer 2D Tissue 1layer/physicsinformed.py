import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from utilities import get_derivative

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedContinuous:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, layers, t0,x0,z0,tb,xb,zb, t_xz_lb,x_xz_lb,z_xz_lb, t_int,x_int, z_int_l3,t_f, x_f, z_f,u_c, Q1):
        """Construct a PhysicsInformedBar model"""

        self.u_c = u_c # constant temp at the surface
        self.t0 = t0
        self.x0 = x0
        self.z0 = z0
        self.tb = tb
        self.xb = xb
        self.zb = zb
        self.t_xz = t_xz_lb
        self.x_xz = x_xz_lb
        self.z_xz = z_xz_lb 
        self.t_bottom =  t_int
        self.t_f = t_f 
        self.x_bottom = x_int 
        self.z_bottom =  z_int_l3
        self.x_f = x_f 
        self.z_f =  z_f
        self.Q1 = Q1
        
        self.l1 = 0.08 # thickness of the layer(m)
        self.rho1 = 0.0012 # density of the layer (kg/m^3)
        self.C1 = 3.6 # specific heat (J/kg. deg Celsius)
        self.Cb1 = 4.2 # specific heat of the blood (J/kg. deg Celsius)
        self.K1 = 0.00026 # thermal conductivity of the tissue (W/m. deg Celsius)
        self.Wb1 = 0 # Blood perfusion rate (g/mm^3. s)
        self.alpha1 = 0.1 #laser absorbtivity of the first layer
        self.Reff1 = 0.93 # Laser reflectivity of the first layer
        
        self.l2 = 2 # thickness of the layer(m)
        self.rho2 = 0.0012 # density of the layer (kg/m^3)
        self.C2 = 3.4 # specific heat (J/kg. deg Celsius)
        self.Cb2 = 4.2 # specific heat of the blood (J/kg. deg Celsius)
        self.K2 = 0.00052 # thermal conductivity of the tissue (W/m. deg Celsius)
        self.Wb2 = 5 * 10**(-7) # Blood perfusion rate (kg/m^3. s)
        self.alpha2 = 0.08 #laser absorbtivity of the second layer
        self.Reff2 = 0.93 # Laser reflectivity of the second layer 
        
        self.l3 = 10 # thickness of the layer(m)
        self.rho3 = 0.001 # density of the layer (kg/m^3)
        self.C3 = 3.06 # specific heat (J/kg. deg Celsius)
        self.Cb3 = 4.2 # specific heat of the blood (J/kg. deg Celsius)
        self.K3 = 0.00021 # thermal conductivity of the tissue (W/m. deg Celsius)
        self.Wb3 = 5 * 10**(-7) # Blood perfusion rate (kg/m^3. s)
        self.alpha3 = 0.04 #laser absorbtivity of the third layer
        self.Reff3 = 0.93 # Laser reflectivity of the third layer 
        
        self.P = 6.4
       
        self.model = self.build_model(layers[0], layers[1:-1], layers[-1])
        self.train_cost_history = []
        

    def build_model(self, input_dimension, hidden_dimension, output_dimension):
        """Build a neural network of given dimensions."""

        nonlinearity = torch.nn.Tanh()
        modules = []
        modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0]))
        modules.append(nonlinearity)
        for i in range(len(hidden_dimension)-1):
            modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i+1]))
            modules.append(nonlinearity)

        modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))

        model = torch.nn.Sequential(*modules).to(device)
        print(model)
        print('model parameters on gpu:', next(model.parameters()).is_cuda)
        return model

    def u_nn(self, t, x , z):
        """Predict temperature at (t,x,z)."""

        u = self.model(torch.cat((t,x,z),1))
        return u

    def f_nn(self, t, x , z ):
        
        """Compute differential equation -> Pennes heat equation"""

        u = self.u_nn(t, x, z)
        u_t = get_derivative(u, t, 1)
        u_x = get_derivative(u, x, 1)
        
        u_z = get_derivative(u, z, 1)
        u_xx = get_derivative(u, x, 2)
        u_zz = get_derivative(u, z, 2)
        
        
        f = self.rho2*self.C2*u_t - self.K2*(u_xx+u_zz) + self.Wb2*self.Cb2*u 
        
        return f

    def cost_function(self):
        """Compute cost function."""
        
        
        u0_pred = self.u_nn(self.t0, self.x0, self.z0)
        
        # initial condition loss @ t = 0  
        
        mse_0 = torch.mean((u0_pred)**2)
        
        # surface boundary condition loss @ z = 0  
        
        u_b_pred = self.u_nn(self.tb, self.xb, self.zb)
        
        mse_b = torch.mean((u_b_pred-self.u_c)**2) 
        
        ##  surface of the tissue loss condition 
        #for xz plane of the tissue
        
        u_xz_pred = self.u_nn(self.t_xz, self.x_xz, self.z_xz) #wall 1
        u_x_xz_pred = get_derivative(u_xz_pred, self.x_xz, 1)
        
        mse_b+= torch.mean(u_x_xz_pred**2)
        
        # for bottom of the tissue
        u_bottom_pred = self.u_nn(self.t_bottom, self.x_bottom, self.z_bottom) #wall 1
        u_z_bottom_pred = get_derivative(u_bottom_pred, self.z_bottom, 1)
        
        mse_b+= torch.mean(u_z_bottom_pred**2)
        
        # for the function loss
        f_pred = self.f_nn(self.t_f,self.x_f,self.z_f)
        mse_f = torch.mean((f_pred)**2)  

        return 1e1*mse_0, 1e1*mse_b, 1e7*mse_f

    def train(self, epochs, optimizer='Adam', **kwargs):
        """ Train the model """

        # Select optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)

        ########################################################################
        elif optimizer=='L-BFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters())

            def closure():
                self.optimizer.zero_grad()
                mse_0, mse_b, mse_f = self.cost_function()
                cost = mse_0 + mse_b + mse_f
                cost.backward(retain_graph=True)
                return cost
        ########################################################################

        # Training loop
        for epoch in range(epochs):
            mse_0, mse_b, mse_f = self.cost_function()
            cost = mse_0 + mse_b + mse_f
            self.train_cost_history.append([cost.cpu().detach(), mse_0.cpu().detach(), mse_b.cpu().detach(), mse_f.cpu().detach()])

            if optimizer=='Adam':
                # Set gradients to zero.
                self.optimizer.zero_grad()

                # Compute gradient (backwardpropagation)
                cost.backward(retain_graph=True)

                # Update parameters
                self.optimizer.step()

            ########################################################################
            elif optimizer=='L-BFGS':
                self.optimizer.step(closure)
            ########################################################################

            if epoch % 100 == 0:
                # print("Cost function: " + cost.detach().numpy())
                print(f'Epoch ({optimizer}): {epoch}, Cost: {cost.detach().cpu().numpy()}, Bound_loss: {mse_b.detach().cpu().numpy()}, Fun_loss: {mse_f.detach().cpu().numpy()}, Ini_loss: {mse_0.detach().cpu().numpy()}')

    def plot_training_history(self, yscale='log'):
        """Plot the training history."""

        train_cost_history = np.asarray(self.train_cost_history, dtype=np.float32)

        # Set up plot
        fig, ax = plt.subplots(figsize=(4,3))
        ax.set_title("Cost function history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function C")
        plt.yscale(yscale)

        # Plot data
        mse_0, mse_b, mse_f = ax.plot(train_cost_history[:,1:4])
        mse_0.set(color='r', linestyle='dashed', linewidth=2)
        mse_b.set(color='k', linestyle='dotted', linewidth=2)
        mse_f.set(color='silver', linewidth=2)
        plt.legend([mse_0, mse_b, mse_f], ['MSE_0', 'MSE_b', 'MSE_f'], loc='lower left')
        plt.tight_layout()
        plt.savefig('cost-function-history.eps')
        plt.show()






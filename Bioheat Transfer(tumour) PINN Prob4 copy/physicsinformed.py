import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from utilities import get_derivative

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedContinuous:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self,layers,t_0,x_0,z_0,t_top,x_top,z_top,t_bott,x_bott,z_bott,
                                      t_left,x_left,z_left,t_right,x_right,z_right,t_f,x_f,z_f,
                                      t_f_t,x_f_t,z_f_t,u_a,Qm,Qm_t):
        """Construct a PhysicsInformedBar model"""

        self.u_a = u_a # ini cond
        self.t0 = t_0
        self.x0 = x_0
        self.z0 = z_0
        self.t_top = t_top
        self.x_top = x_top
        self.z_top = z_top
        self.t_bott = t_bott
        self.x_bott = x_bott
        self.z_bott = z_bott 
        self.t_left =  t_left
        self.x_left = x_left
        self.z_left =  z_left
        self.t_right = t_right
        self.x_right = x_right
        self.z_right = z_right
        self.t_f = t_f 
        self.x_f = x_f 
        self.z_f =  z_f
        self.t_f_t = t_f_t 
        self.x_f_t = x_f_t
        self.z_f_t =  z_f_t
        self.Qm = Qm
        self.Qm_t = Qm_t
        
        self.rho = 1000 # density of the layer (kg/m^3)
        self.rhob = 1000 # density of the blood vessel (kg/m^3)
        self.C = 4000 # specific heat (J/kg. deg Celsius)
        self.Cb = 4000 # specific heat of the blood (J/kg. deg Celsius)
        self.K = 0.5 # thermal conductivity of the tissue (W/m. deg Celsius)
        self.Wb = 0.0005 # Blood perfusion rate (m^3/s/m^3)
        self.Wb_t = 0.002 # Blood perfusion rate in tumour (m^3/s/m^3)
        self.depth = 0.017
        self.length = 0.017
        self.Q = 0
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

    def u_nn(self, t, x1 , x2):
        """Predict temperature at (t,x,z)."""

        u = self.model(torch.cat((t,x1,x2),1))
        return u

    def f_nn(self, t, x1 , x2, i=None ):
        
        """Compute differential equation -> Pennes heat equation"""

        u = self.u_nn(t, x1, x2)
        u_t = get_derivative(u, t, 1)
        
        u_xx1 = get_derivative(u, x1, 2)
        u_xx2 = get_derivative(u, x2, 2)
        if i==1: # for non tumor region
            Wb = self.Wb
            Qm = self.Qm
        else:
            Wb = self.Wb_t
            Qm = self.Qm_t
        
        f = self.rho*self.C*u_t - self.K*(u_xx1+u_xx2) - Wb*self.rhob*self.Cb*(self.u_a-u)-Qm-self.Q 
        
        return f

    def cost_function(self):
        """Compute cost function."""
        
        
        u0_pred = self.u_nn(self.t0, self.x0, self.z0)
        
        # initial condition loss @ t = 0  
        
        mse_0 = torch.mean((u0_pred)**2)
        
        # boundary condition loss top, bottom, left and right
        
        u_top_pred = self.u_nn(self.t_top, self.x_top, self.z_top)
        
        mse_b = torch.mean((u_top_pred-self.u_a)**2) 
        
        u_bott_pred = self.u_nn(self.t_bott, self.x_bott, self.z_bott)
        u_bott_pred_x = get_derivative(u_bott_pred, self.x_bott, 1)
        u_bott_pred_z = get_derivative(u_bott_pred, self.z_bott, 1)
        
        mse_b+= torch.mean((u_bott_pred_x**2+u_bott_pred_z**2))
        
        u_left_pred = self.u_nn(self.t_left, self.x_left, self.z_left)
        u_left_pred_x = get_derivative(u_left_pred, self.x_left, 1)
        u_left_pred_z = get_derivative(u_left_pred, self.z_left, 1)
        
        mse_b+= torch.mean((u_left_pred_x**2+u_left_pred_z**2))
        
        u_right_pred = self.u_nn(self.t_right, self.x_right, self.z_right)
        u_right_pred_x = get_derivative(u_right_pred, self.x_right, 1)
        u_right_pred_z = get_derivative(u_right_pred, self.z_right, 1)
        
        mse_b+= torch.mean((u_right_pred_x**2+u_right_pred_z**2))
        
        # for the function loss non tumor region
        f_pred = self.f_nn(self.t_f,self.x_f,self.z_f,1)
        mse_f = torch.mean((f_pred)**2)  
        # for the function loss tumor region
        f_pred_t = self.f_nn(self.t_f_t,self.x_f_t,self.z_f_t)
        mse_f += torch.mean((f_pred_t)**2)  

        return mse_0, 1e-3*mse_b, 1e-9*mse_f

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






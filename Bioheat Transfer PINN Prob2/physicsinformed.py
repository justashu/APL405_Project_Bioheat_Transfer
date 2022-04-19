import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from utilities import get_derivative

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedContinuous:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, layers, t0,x0,z0,tb,xb,zb, t_xz_lb,x_xz_lb,z_xz_lb, t_int,x_int, z_int_l3,t_f, x_f, z_f,u_a, Q1,Qm):
        """Construct a PhysicsInformedBar model"""

        self.u_a = u_a # temp of the vessel
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
        self.Qm = Qm
        self.Q1 = Q1
        
        self.rho1 = 1000 # density of the layer (kg/m^3)
        self.rhob = 1000 # density of the blood vessel (kg/m^3)
        self.C1 = 4000 # specific heat (J/kg. deg Celsius)
        self.Cb1 = 4000 # specific heat of the blood (J/kg. deg Celsius)
        self.K1 = 0.5 # thermal conductivity of the tissue (W/m. deg Celsius)
        self.Wb1 = Wb1 = 0.0005 # Blood perfusion rate (m^3/s/m^3)
        self.depth = 0.017
        self.length = 0.017
       
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

    def f_nn(self, t, x1 , x2 ):
        
        """Compute differential equation -> Pennes heat equation"""

        u = self.u_nn(t, x1, x2)
        u_t = get_derivative(u, t, 1)
        
        u_xx1 = get_derivative(u, x1, 2)
        u_xx2 = get_derivative(u, x2, 2)
        
        
        f = self.rho1*self.C1*u_t - self.K1*(u_xx1+u_xx2) - self.Wb1*self.rhob*self.Cb1*(self.u_a-u)-self.Qm-self.Q1(t,x1,x2) 
        
        return f

    def cost_function(self):
        """Compute cost function."""
        
        
        u0_pred = self.u_nn(self.t0, self.z0, self.x0)
        
        # initial condition loss @ t = 0  
        
        mse_0 = torch.mean((u0_pred-(torch.cos(torch.pi*self.z0)+torch.sin(torch.pi*self.x0)+torch.tanh(self.z0+self.x0)+37))**2)
        
        # surface boundary condition loss @ z = 0  
        
        u_b_pred = self.u_nn(self.tb, self.zb, self.xb)
        
        mse_b = torch.mean((u_b_pred-(torch.exp(-self.tb)*(1+torch.sin(torch.pi*self.xb))+torch.tanh(self.xb)+37))**2) 
        
        u_z_b_pred = get_derivative(u_b_pred, self.zb, 1)
        mse_b+= torch.mean((u_z_b_pred-(1/(torch.cosh(self.xb))**2))**2)
        
        # @ z = depth(bottom)
        zbb = self.zb*0+self.depth
        u_b_pred_bot = self.u_nn(self.tb, zbb, self.xb)
        u_z_b_pred_bot = get_derivative(u_b_pred_bot, zbb, 1)
        
        mse_b += torch.mean((u_b_pred_bot-(torch.exp(-self.tb)*(torch.cos(torch.pi*zbb)+torch.sin(torch.pi*self.xb))+torch.tanh(self.xb+zbb)+37))**2)
        
        mse_b+= torch.mean((u_z_b_pred_bot-(torch.exp(-self.tb)*(-1)*torch.pi*(torch.sin(torch.pi*zbb))+(1/torch.cosh(self.xb+zbb))**2))**2)
        
        ##  right bound (x2=length)of the tissue loss condition 
        x_xzb = self.x_xz*0+self.length
        u_xz_predr = self.u_nn(self.t_xz, self.z_xz, x_xzb) #wall 1
        u_x_xz_predr = get_derivative(u_xz_predr, x_xzb, 1)
        
        mse_b+= torch.mean((u_x_xz_predr-(torch.exp(-self.t_xz)*torch.pi*(torch.cos(torch.pi*x_xzb))+(1/torch.cosh(self.z_xz+x_xzb))**2))**2)
        
        ##  left bound (x2=0)of the tissue loss condition 
        x_xz = self.x_xz*0
        u_xz_predl = self.u_nn(self.t_xz, self.z_xz, x_xz) #wall 1
        u_x_xz_predl = get_derivative(u_xz_predl, x_xz, 1)
        
        mse_b+= torch.mean((u_x_xz_predl-(torch.exp(-self.t_xz)*torch.pi*(torch.cos(torch.pi*x_xz))+(1/torch.cosh(self.z_xz+x_xz))**2))**2)
        
        # for bottom of the tissue
        #u_bottom_pred = self.u_nn(self.t_bottom, self.x_bottom, self.z_bottom) #wall 1
        #u_z_bottom_pred = get_derivative(u_bottom_pred, self.z_bottom, 1)
        
        #mse_b+= torch.mean(u_z_bottom_pred**2)
        
        # for the function loss
        f_pred = self.f_nn(self.t_f,self.z_f,self.x_f)
        mse_f = torch.mean((f_pred)**2)  

        return mse_0, mse_b, 1e-7*mse_f

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






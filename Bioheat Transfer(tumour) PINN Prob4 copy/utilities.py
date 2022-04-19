import torch
from torch.autograd import grad

'''import matplotlib
import matplotlib.font_manager
matplotlib.rcParams["figure.dpi"] = 80
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],
             'size' : 10})
rc('text', usetex=True)
'''
dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_derivative( y, x, n):
    """Compute the n-th order derivative of y = f(x) with respect to x."""
    if n == 0:
        return y
    else:
        dy_dx = grad(y, x, torch.ones_like(y).to(device), create_graph=True, retain_graph=True, allow_unused=True)[0]
        return get_derivative(dy_dx, x, n - 1)


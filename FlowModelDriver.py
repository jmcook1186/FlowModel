import numpy as np
from ModelFuncs import TransientFlowModel, vector_arrows
import matplotlib.pyplot as plt

x = np.arange(-500., 500., 10.)
y = np.arange(-500., 500., 10.)# backward, i.e. first row grid line has highest y
z = np.arange(0., 2., 0.2)# backward, i.e. from top to bottom
SHP = (len(z)-1, len(y)-1, len(x)-1)

k = 0.001 # m/d uniform conductivity

kx = k*np.ones(SHP)# [L/T] 3D kx array
ky = k*np.ones(SHP)# [L/T] 3D ky array with same values as kx
kz = k*np.ones(SHP)# [L/T] 3D kz array with same values as kx

FQ = np.zeros(SHP) + 2 # all flows zero. Note sz is the shape of the model grid
FQ[:, 0:-1, 20:25] = -200 # [m3/d] extraction in this cell

HI = np.random.rand(len(z)-1,len(y)-1,len(x)-1)*110

IBOUND = np.ones(SHP)
IBOUND[:, -1, :] = -1 # last row of model heads are prescribed
#IBOUND[:, 40:45, 20:70]=0 # these cells are inactive

t = np.arange(0,10,1)
Ss = 0.01 # specific storage term

Out = TransModel(x, y, z, t, kx, ky, kz, Ss, FQ, HI, IBOUND, epsilon=0.67)


print('Out.Phi.shape ={0}'.format(Out.Phi.shape))
print('Out.Q.shape ={0}'.format(Out.Q.shape))
print('Out.Qx.shape ={0}'.format(Out.Qx.shape))
print('Out.Qy.shape ={0}'.format(Out.Qy.shape))
print('Out.Qz.shape ={0}'.format(Out.Qz.shape))



# xm = 0.5*(x[:-1] + x[1:])
# ym = 0.5*(y[:-1] + y[1:])
# layer = 0# contours for this layer
# nc = 50 # number of contours in total
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.title("Contours ({} in total) of the head in layer {} with inactive section".format(nc, layer))
# plt.contour(xm, ym, Out.Phi[-1,layer,:,:], nc),plt.colorbar()

# X, Y, U, V = vector_arrows(Out, x, y, iz=0)
# plt.quiver(X, Y, U, V)


for i in t:

    plt.figure()
    plt.title('Q of cells in bottom layer with well [m3/d]')
    plt.imshow(Out.Q[i,0, :, :], vmin=-5,vmax=5)
    plt.colorbar()
    plt.savefig('/home/joe/Code/FlowModel/FDM3_{}.png'.format(i))
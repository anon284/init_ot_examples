# definte ot problem
import os, sys
sys.path.append('./ott')
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from ott.core.sinkhorn import sinkhorn
from ott.geometry.pointcloud import PointCloud
import jax
from jax import numpy as jnp
import ott
from ott.core import initializers as init_lib
import numpy as np
import matplotlib.pyplot as plt
from ott.core.sinkhorn import sinkhorn
from ott.geometry.pointcloud import PointCloud
import jax
from jax import numpy as jnp
import ott
from ott.core import initializers as init_lib


def gaus_ot(seed=0):
  np.random.seed(seed)
  n, d = 1000, 2
  mu_a = np.array([-1,1])*5
  mu_b = np.array([0,0])


  x = np.random.normal(size=n*d).reshape(n,d) + mu_a
  y = np.random.normal(size=n*d).reshape(n,d) + mu_b


  n = len(x)
  m = len(y)
  a = np.ones(n)/n
  b = np.ones(m)/m

  x_jnp, y_jnp = jnp.array(x), jnp.array(y)
  return x_jnp, y_jnp, a, b

x, y, a, b = gaus_ot()  

plt.plot(x[:,0], x[:,1], 'rx')
plt.plot(y[:,0], y[:,1], 'bx')
plt.show()

gaus_init = init_lib.GaussianInitializer()

@jax.jit
def run_sinkhorn_gaus_init(x, y, a=None, b=None, init_dual_a=None):
    sink_kwargs = {'jit': True, 
                   'threshold': 0.001, 
                   'max_iterations': 10**5, 
                   'potential_initializer': gaus_init}
                   
    geom_kwargs = {'epsilon': 0.01}
    geom = PointCloud(x, y, **geom_kwargs)
    out = sinkhorn(geom, a=a, b=b, init_dual_a=init_dual_a, **sink_kwargs)
    return out

@jax.jit
def run_sinkhorn(x, y, a=None, b=None, init_dual_a=None):
    sink_kwargs = {'jit': True, 'threshold': 0.001, 'max_iterations': 10**5}
    geom_kwargs = {'epsilon': 0.01}
    geom = PointCloud(x, y, **geom_kwargs)
    out = sinkhorn(geom, a=a, b=b, init_dual_a=init_dual_a, **sink_kwargs)
    return out
    

x, y, a, b = gaus_ot(seed=0) 

default_sink_out = run_sinkhorn(x=x,y=y,a=a, b=b)
sink_f = default_sink_out.f 
sink_g = default_sink_out.g
num_iter = jnp.sum(default_sink_out.errors > -1)
print(f'0 Init, num iterations: {num_iter}')

sink_out = run_sinkhorn_gaus_init(x=x,y=y, a=a, b=b)
sink_f = sink_out.f 
sink_g = sink_out.g
num_iter = jnp.sum(sink_out.errors > -1)
print(f'Gaus Init, num iterations: {num_iter}')


x, y, a, b = gaus_ot(seed=1) 

default_sink_out = run_sinkhorn(x=x,y=y,a=a, b=b, init_dual_a=default_sink_out.f)
sink_f = default_sink_out.f 
sink_g = default_sink_out.g
num_iter = jnp.sum(default_sink_out.errors > -1)
print(f'Warm start Init, num iterations: {num_iter}')

sink_out = run_sinkhorn_gaus_init(x=x,y=y, a=a, b=b)
sink_f = sink_out.f 
sink_g = sink_out.g
num_iter = jnp.sum(sink_out.errors > -1)
print(f'Gaus Init, num iterations: {num_iter}')

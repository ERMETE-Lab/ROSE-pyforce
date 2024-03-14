# Fundamental tools
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 30 August 2023
# Latest Doc  Update: 25 June  2023

from importlib import metadata
import numpy as np
import time
import warnings

# import dolfinx
from dolfinx.fem import (Function, FunctionSpace, assemble_scalar, assemble_vector, form, Expression)
from ufl import grad, inner, dx, Measure
from ufl.domain import extract_unique_domain
# Note that dx will be deprecated in the future, is it necessary to import the mesh in addition to V?

# Class to make progress bar
class LoopProgress():
    r"""
    A class to make progress bar.

    Parameters
    ----------
    msg : str
        Message to be displayed
    final : float, optional (Default = 100)
        Maximum value for the iterations

    """
    def __init__(self, msg: str, final: float = 100):
        self.msg = msg
        self.final = final
        self.instant = 0.

        self.init_time  = time.time()
        self.comp_times = list()

        out =  self.msg+': '
        print (out, end="\r")

    def update(self, step: float, percentage: bool = False):
        r"""
        Update message to display and clears the previous one.
        
        Parameters
        ----------
        step : float
            Interger or float value to add at the counter.
        percentage : boolean, optional (Default = False)
            Indicates if the bar should be displayed in %.
        
        """

        # Compute average computational time
        self.comp_times.append(time.time() - self.init_time)        
        average_time = sum(self.comp_times) / len(self.comp_times)

        # Update instant
        self.instant += step

        # Write the message
        if percentage:
            printed_inst = '{:.3f}'.format(self.instant / self.final * 100)+' / 100.00%'
        else:
            printed_inst = '{:.3f}'.format(self.instant)+' / {:.2f}'.format(self.final)
        out =  self.msg+': '+printed_inst + ' - {:.3f}'.format(average_time)+' s/it'

        # Print output
        if np.isclose(self.instant, self.final):
            print (out)
        else:
            print (out, end="\r")

        # Update inital offset cpu time
        self.init_time  = time.time()

# Class to compute norms in L2, H1 and L^\infty and the inner product in L2
class norms():
    r"""
        A class to compute norms and inner products. :math:`L^2` and :math:`H^1` (semi and full are implemented for both scalar and vector fields), whereas the average and the integral are available for scalar only.

        Parameters
        ----------
        V : FunctionSpace
            Functional Space onto which the Function are defined.
        is_H1 : boolean, optional (Default = False)
            If the function belongs to :math:`H^1`, the forms for the inner products and norms are computed.

    """
    def __init__(self, V: FunctionSpace, is_H1 = False, metadata_degree=4):
        
        self.V = V
        self.u1 = Function(V).copy()
        self.u2 = Function(V).copy()
        
        # Deprecation warning in fenics-dolfinx=0.6.0 --> try to correct this?
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        
        metadata = {"quadrature_degree": metadata_degree}
        self.dx = Measure("dx", domain=extract_unique_domain(self.u1), metadata=metadata)

        if V.num_sub_spaces == 0: # if the functional space is related to a scalar function
            self.integ_form = form(self.u1 * self.dx) 

        self.L2form_inner = form(inner(self.u1, self.u2) * self.dx)
        
        if is_H1:
            self.semiH1form_inner = form(inner(grad(self.u1), grad(self.u2)) * self.dx)
            self.fullH1form_inner = form( (inner(grad(self.u1), grad(self.u2)) + inner(self.u1, self.u2)) * self.dx)
            
    def integral(self, u: Function):
        r""" 
        Computes the integral of a given scalar function `u` over the domain

        .. math::
            \int_\Omega u \,d\Omega 

        Parameters
        ----------
        u : Function
            Function belonging to the same functional space V (it must be a scalar!)

        Returns
        -------
        val : float
            Integral over the domain
        """
        
        if not isinstance(u, Function):
            if isinstance(u, np.ndarray) and ( self.u1.x.array.shape == u.shape ):
                self.u1.x.array[:] = u
            else:
                tmp = Function(self.V).copy()
                tmp.interpolate(Expression(u, self.V.element.interpolation_points()))
                warnings.warn("Input u is not a function: it may slower the calculations")            

                self.u1.x.array[:] = tmp.x.array[:]
        else:
            self.u1.x.array[:] = u.x.array[:]
            
        val = assemble_scalar(self.integ_form)
            
        return val

    def average(self, u: Function):
        r""" 
        Computes the integral average of a given **scalar** function `u` over the domain

        .. math::
            \langle u \rangle = \frac{1}{|\Omega|}\int_\Omega u \,d\Omega

        Parameters
        ----------
        u : Function
            Function belonging to the same functional space V (it must be a scalar!)

        Returns
        -------
        ave_value : float
            Average over the domain
        """
        
        value = self.integral(u)
        
        dom_fun = Function(self.V).copy()
        dom_fun.x.set(1.0)
        domain_norm = self.integral(dom_fun)
        
        ave_value = value / domain_norm
        return ave_value

    def L2norm(self, u: Function):
        r""" 
        Computes the :math:`L^2` norm of the function `u` over the domain

        .. math::
            \| u\|_{L^2} = \sqrt{\int_\Omega u \cdot u\,d\Omega}

        Parameters
        ----------
        u : Function
            Function belonging to the same functional space `V`

        Returns
        -------
        value : float
            :math:`L^2` norm of the function
        """

        if not isinstance(u, Function):
            if isinstance(u, np.ndarray) and ( self.u1.x.array.shape == u.shape ):
                self.u1.x.array[:] = u
                self.u2.x.array[:] = u
            else:
                tmp = Function(self.V).copy()
                tmp.interpolate(Expression(u, self.V.element.interpolation_points()))
                warnings.warn("Input u is not a function: it may slower the calculations")            

                self.u1.x.array[:] = tmp.x.array[:]
                self.u2.x.array[:] = tmp.x.array[:]
        else:
            self.u1.x.array[:] = u.x.array[:]
            self.u2.x.array[:] = u.x.array[:]

        value = np.sqrt(assemble_scalar(self.L2form_inner))
            
        return value
                       
    def H1norm(self, u: Function, semi = True):
        r""" 
        Computes the :math:`H^1` semi or full norm of the function `u` over the domain

        .. math::
            | u |_{H^1} = \sqrt{\int_\Omega \nabla u \cdot \nabla u\,d\Omega}

            
        .. math::
            \| u \|_{H^1} = \sqrt{\int_\Omega \nabla u \cdot \nabla u\,d\Omega + \int_\Omega u \cdot  u\,d\Omega}

        Parameters
        ----------
        u : Function
            Function belonging to the same functional space `V`
        semi : boolean, optional (Default = True)
            Indicates if the semi norm must be computed.
        
        Returns
        -------
        value : float
            :math:`H^1` norm of the function
        """

        if not isinstance(u, Function):
            if isinstance(u, np.ndarray) and ( self.u1.x.array.shape == u.shape ):
                self.u1.x.array[:] = u
                self.u2.x.array[:] = u
            else:
                tmp = Function(self.V).copy()
                tmp.interpolate(Expression(u, self.V.element.interpolation_points()))
                warnings.warn("Input u is not a function: it may slower the calculations")            

                self.u1.x.array[:] = tmp.x.array[:]
                self.u2.x.array[:] = tmp.x.array[:]
        else:
            self.u1.x.array[:] = u.x.array[:]
            self.u2.x.array[:] = u.x.array[:]

        if semi == True:
            value = np.sqrt(assemble_scalar(self.semiH1form_inner))
        else:
            value = np.sqrt(assemble_scalar(self.fullH1form_inner))
        
        return value
    
    def Linftynorm(self, u: Function):
        r""" 
        Computes the :math:`L^\infty` norm of a given function `u` over the domain

        .. math::
            \| u \|_{L^\infty}=\max\limits_\Omega |u|

        Parameters
        ----------
        u : Function
            Function belonging to the same functional space `V`

        Returns
        -------
        value : float
            :math:`L^\infty` norm of the function
        """

        if not isinstance(u, Function):
            if isinstance(u, np.ndarray) and ( self.u1.x.array.shape == u.shape ):
                return np.max(np.abs(u))
            else:
                tmp = Function(self.V).copy()
                tmp.interpolate(Expression(u, self.V.element.interpolation_points()))
                warnings.warn("Input u is not a function: it may slower the calculations")            

                value = np.max(np.abs(tmp.x.array)) 
        else:
            value = np.max(np.abs(u.x.array))
            
        return value


    def L2innerProd(self, u: Function, v: Function):
        r""" 
        Computes the :math:`L^2` inner product of the functions `u` and `v` over the domain

        .. math::
            (u,v)_{L^2}=\int_\Omega u\cdot v \,d\Omega

        Parameters
        ----------
        u : Function
            Function belonging to the same functional space `V`
        v : Function
            Function belonging to the same functional space `V`

        Returns
        -------
        value : float
            :math:`L^2` inner product between the functions
        """

        if not isinstance(u, Function):
            if isinstance(u, np.ndarray) and ( self.u1.x.array.shape == u.shape ):
                self.u1.x.array[:] = u
            else:
                tmp = Function(self.V).copy()
                tmp.interpolate(Expression(u, self.V.element.interpolation_points()))
                warnings.warn("Input u is not a function: it may slower the calculations")            

                self.u1.x.array[:] = tmp.x.array[:]
        else:
            self.u1.x.array[:] = u.x.array[:]

        if not isinstance(v, Function):
            if isinstance(v, np.ndarray) and ( self.u2.x.array.shape == v.shape ):
                self.u2.x.array[:] = v
            else:
                tmp = Function(self.V).copy()
                tmp.interpolate(Expression(v, self.V.element.interpolation_points()))
                warnings.warn("Input u is not a function: it may slower the calculations")            

                self.u2.x.array[:] = tmp.x.array[:]
        else:
            self.u2.x.array[:] = v.x.array[:]

        value = assemble_scalar(self.L2form_inner)
        return value
                       
    def H1innerProd(self, u: Function, v: Function, semi = True):
        r""" 
        Computes the :math:`H^1` semi or full inner product of the functions `u` and `v` over the domain

        .. math::
            \langle u, v \,\rangle_{H^1} = \int_\Omega \nabla u \cdot \nabla v\,d\Omega

            
        .. math::
            (u,v)_{H^1} = \int_\Omega u\cdot v \,d\Omega + \int_\Omega \nabla u\cdot \nabla v \,d\Omega

        Parameters
        ----------
        u : Function
            Function belonging to the same functional space `V`
        v : Function
            Function belonging to the same functional space `V`
        semi : boolean, optional (Default = True)
            Indicates if the semi norm must be computed.
        
        Returns
        -------
        value : float
            :math:`H^1` inner product of the functions
        """

        if not isinstance(u, Function):
            if isinstance(u, np.ndarray) and ( self.u1.x.array.shape == u.shape ):
                self.u1.x.array[:] = u
            else:
                tmp = Function(self.V).copy()
                tmp.interpolate(Expression(u, self.V.element.interpolation_points()))
                warnings.warn("Input u is not a function: it may slower the calculations")            

                self.u1.x.array[:] = tmp.x.array[:]
        else:
            self.u1.x.array[:] = u.x.array[:]

        if not isinstance(v, Function):
            if isinstance(v, np.ndarray) and ( self.u2.x.array.shape == v.shape ):
                self.u2.x.array[:] = v
            else:
                tmp = Function(self.V).copy()
                tmp.interpolate(Expression(v, self.V.element.interpolation_points()))
                warnings.warn("Input u is not a function: it may slower the calculations")            

                self.u2.x.array[:] = tmp.x.array[:]
        else:
            self.u2.x.array[:] = v.x.array[:]

        if semi == True:
            value = assemble_scalar(self.semiH1form_inner)
        else:
            value = assemble_scalar(self.fullH1form_inner)

        return value
    
    
# class compute_integral():
#     def __init__(self, domain, V):
#         self.domain = domain
#         self.V = V
        
#         v = TestFunction(V)
#         cell_area_form = form(v*dx)
#         self.cell_area = assemble_vector(cell_area_form).array
        
#         self.h = dolfinx.cpp.mesh.h(domain, 2, range(len(domain.geometry.x)))
        
#     def integral(self,u):
#         output = sum(u.x.array[:] * self.cell_area)
#         return output
    
#     def L2norm(self,u):
#         output = np.sqrt(sum(u.x.array[:]**2 * self.cell_area))
#         return output
# Fundamental tools
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 01 November 2024
# Latest Doc  Update: 01 November 2024

import numpy as np
import time
import warnings

from dolfinx.fem import (Function, FunctionSpace, assemble_scalar, assemble_vector, assemble_matrix, form, Expression, petsc)
from ufl import grad, inner, Measure
from ufl.domain import extract_unique_domain

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

        # Definition of the variational forms
        if V.num_sub_spaces == 0: # if the functional space is related to a scalar function
            self.integ_form = form(self.u1 * self.dx) 

        self.L2form_inner = form(inner(self.u1, self.u2) * self.dx)
        
        if is_H1:
            self.semiH1form_inner = form(inner(grad(self.u1), grad(self.u2)) * self.dx)
            self.fullH1form_inner = form( (inner(grad(self.u1), grad(self.u2)) + inner(self.u1, self.u2)) * self.dx)
            
    def check_input(self, input: Function):
        r"""
        Check that the input is either a Function or a numpy array of the right shape.
        If not, the code will probably produce an error.

        This method is meant to be used internally.
        """

        if isinstance(input, Function):
            _u  = input.x.array[:]
        else:
            _u = input

        return _u
    
    def integral(self, u: Function):
        r""" 
        Computes the integral of a given scalar function `u` over the domain

        .. math::
            \int_\Omega u \,d\Omega 

        Parameters
        ----------
        u : `Function` (or `np.ndarray`)
            Function belonging to the same functional space V (it must be a scalar!)

        Returns
        -------
        value : float
            Integral over the domain
        """
        
        _u = self.check_input(u)
        
        self.u1.x.array[:] = _u
            
        value = assemble_scalar(self.integ_form)
            
        return value

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

        _u = self.check_input(u)
        _v = self.check_input(v)

        self.u1.x.array[:] = _u
        self.u2.x.array[:] = _v

        value = assemble_scalar(self.L2form_inner)
        return value

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
        
        value = np.sqrt(self.L2innerProd(u,u))
            
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

        _u = self.check_input(u)
        _v = self.check_input(v)
        
        self.u1.x.array[:] = _u
        self.u2.x.array[:] = _v

        if semi == True:
            value = assemble_scalar(self.semiH1form_inner)
        else:
            value = assemble_scalar(self.fullH1form_inner)

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

        value = np.sqrt(self.H1innerProd(u,u, semi=semi))

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

        _u = self.check_input(u)

        value = np.max(np.abs(_u))
            
        return value
    
# Class to make progress bar using printing
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

########################################################################


# # Class to compute norms in L2, H1 and L^\infty and the inner product in L2 - using FE representation
# class np_norms():
#     r"""
#         A class to compute norms and inner products. 
#         :math:`L^2` and :math:`H^1` (semi and full are implemented for both scalar and vector fields), whereas the average and the integral are available for scalar only.

#         Given the mesh :math:`T_h`, a proper basis is chosen for this domain :math:`\{\varphi_k(\boldsymbol{x})\}_{k=1}^{N_h}`, where N_h is the number of degrees of freedom.
#         Any function can be represented as a linear combination of the basis functions:

#         .. math::
#             u(\boldsymbol{x}) = \sum_{k=1}^{N_h} a_k \, \varphi_k(\boldsymbol{x})

#         The class creates the mass matrix, assembled from the following bilinear form
        
#         .. math::
#             \mathbb{M}_{ij} = \int_\Omega \varphi_i\cdot \varphi_j \,d\Omega \qquad i,j = 1, \dots, N_h

#         and the stiffness matrix, assembled from the following bilinear form:
        
#         .. math::
#             \mathbb{A}_{ij} = \int_\Omega \nabla\varphi_i\cdot \nabla\varphi_j \,d\Omega \qquad i,j = 1, \dots, N_h

#         and the "right-hand side" vector, assembled from the following linear form:

#         .. math::
#             L_{i} = \int_\Omega \varphi_i \,d\Omega \qquad i = 1, \dots, N_h

#         These matrices are used to compute the norms and inner products, reducing the computational cost.

#         Parameters
#         ----------
#         V : FunctionSpace
#             Functional Space onto which the Function are defined.

#     """
#     def __init__(self, V: FunctionSpace):
        
#         self.V = V
        
#         u = ufl.TrialFunction(V)
#         v = ufl.TestFunction(V)

#         # Assemble mass matrix int_Omega (u * v * dx)
#         mass_form = ufl.inner(u, v) * ufl.dx
#         mass_matrix = petsc.assemble_matrix(form(mass_form))
#         mass_matrix.assemble()

#         self.mass_matrix = np.asarray([mass_matrix.getColumnVector(ii).getArray() for ii in range(mass_matrix.size[0])])

#         # Assemble stiffness matrix int_Omega (\nabla u * \nabla v * dx)
#         stiff_form = ufl.inner(grad(u), grad(v)) * ufl.dx
#         stiff_matrix = petsc.assemble_matrix(form(stiff_form))
#         stiff_matrix.assemble()

#         self.stiff_matrix = np.asarray([stiff_matrix.getColumnVector(ii).getArray() for ii in range(stiff_matrix.size[0])])

#         # Assemble "right-hand size" vec int_Omega (u * dx)
#         rhs_form = u * ufl.dx
#         rhs_vec = petsc.assemble_vector(form(rhs_form))
#         rhs_vec.assemble()

#         self.rhs_vec = rhs_vec.getArray()

#     def check_input(self, input):
#         r"""
#         Check that the input is either a Function or a numpy array of the right shape.
#         If not, the code will probably produce an error.

#         This method is meant to be used internally.
#         """

#         if isinstance(input, Function):
#             _u  = input.x.array[:]
#         else:
#             _u = input

#         return _u

#     def L2innerProd(self, u: Function, v: Function):
#         r""" 
#         Computes the :math:`L^2` inner product of the functions `u` and `v` over the domain

#         .. math::
#             (u,v)_{L^2}=\int_\Omega u\cdot v \,d\Omega

#         Since :math:`u` and :math:`v` over the domain can be expressed as a linear combination of the basis functions, the inner product can be computed as:

#         .. math::
#             (u,v)_{L^2} = \sum_{k=1}^{N_h} \sum_{l=1}^{N_h} a_u^k \cdot a_v^l \, \int_\Omega \varphi_k\cdot \varphi_l \,d\Omega = \boldsymbol{a}_u^T \cdot \mathbb{M} \cdot \boldsymbol{a}_v

#         Parameters
#         ----------
#         u : `Function` (or `np.ndarray`)
#              Function belonging to the same functional space `V`
#         v : `Function` (or `np.ndarray`)
#             Function belonging to the same functional space `V`

#         Returns
#         -------
#         value : float
#             :math:`L^2` inner product between the functions
#         """

#         _u = self.check_input(u)
#         _v = self.check_input(v)

#         value = _u.T @ self.mass_matrix @ _v # np.linalg.multi_dot([_u.T, self.mass_matrix, _v])
#         return value
    
#     def L2norm(self, u: Function):
#         r""" 
#         Computes the :math:`L^2` norm of the function `u` over the domain

#         .. math::
#             \|u\|_{L^2}=\sqrt{\int_\Omega u^2 \,d\Omega}
        
#         The norm is evaluated from the associated scalar product.

#         Parameters
#         ----------
#         u : `Function` (or `np.ndarray`)
#              Function belonging to the same functional space `V`

#         Returns
#         -------
#         value : float
#             :math:`L^2` norm between the functions
#         """

#         _u = self.check_input(u)

#         L2_inner_prod = self.L2innerProd(_u, _u)
#         return np.sqrt(L2_inner_prod)

#     def H1innerProd(self, u: Function, v: Function, semi = True):
#         r""" 
#         Computes the :math:`H^1` (semi or full) inner product of the functions `u` and `v` over the domain

#         .. math::
#             ( u, v)_{H^1_{semi}} = \int_\Omega \nabla u \cdot \nabla v\,d\Omega
            
#         .. math::
#             (u,v)_{H^1} = \int_\Omega u\cdot v \,d\Omega + \int_\Omega \nabla u\cdot \nabla v \,d\Omega

#         Since :math:`u` and :math:`v` over the domain can be expressed as a linear combination of the basis functions, the inner product (semi for instance) can be computed as:

#         .. math::
#             (u,v)_{H^1_{semi}} = \sum_{k=1}^{N_h} \sum_{l=1}^{N_h} a_u^k \cdot a_v^l \, \int_\Omega \nabla\varphi_k\cdot \nabla\varphi_l \,d\Omega = \boldsymbol{a}_u^T \cdot \mathbb{A} \cdot \boldsymbol{a}_v

#         Parameters
#         ----------
#         u : `Function` (or `np.ndarray`)
#              Function belonging to the same functional space `V`
#         v : `Function` (or `np.ndarray`)
#             Function belonging to the same functional space `V`
#         semi : boolean, optional (Default = True)
#             Indicates if the semi norm must be computed.

#         Returns
#         -------
#         value : float
#             :math:`H^1` inner product of the functions
#         """

#         _u = self.check_input(u)
#         _v = self.check_input(v)

#         if semi:
#             value = np.linalg.multi_dot([_u.T, self.stiff_matrix, _u])
#         else:
#             value = np.linalg.multi_dot([_u.T, self.stiff_matrix, _u]) + self.L2innerProd(_u, _v)

#         return value
    
#     def H1norm(self, u: Function, semi = True):
#         r""" 
#         Computes the :math:`H^1` semi or full norm of the function `u` over the domain

#         .. math::
#             | u |_{H^1} = \sqrt{\int_\Omega \nabla u \cdot \nabla u\,d\Omega}

            
#         .. math::
#             \| u \|_{H^1} = \sqrt{\int_\Omega \nabla u \cdot \nabla u\,d\Omega + \int_\Omega u \cdot  u\,d\Omega}

#         The norm is evaluated from the associated scalar product.

#         Parameters
#         ----------
#         u : `Function` (or `np.ndarray`)
#              Function belonging to the same functional space `V`
#         semi : boolean, optional (Default = True)
#             Indicates if the semi norm must be computed.
        
#         Returns
#         -------
#         value : float
#             :math:`H^1` norm of the function
#         """

#         _u = self.check_input(u)

#         H1_inner_prod = self.H1innerProd(_u, _u, semi=semi)

#         return np.sqrt(H1_inner_prod)
    
#     def Linftynorm(self, u: Function):
#         r""" 
#         Computes the :math:`L^\infty` norm of a given function `u` over the domain

#         .. math::
#             \| u \|_{L^\infty}=\max\limits_\Omega |u|

#         Parameters
#         ----------
#         u : `Function` (or `np.ndarray`)
#             Function belonging to the same functional space `V`

#         Returns
#         -------
#         value : float
#             :math:`L^\infty` norm of the function
#         """

#         _u = self.check_input(u)

#         value = np.max(np.abs(_u))
            
#         return value
    
#     def integral(self, u: Function):
#         r""" 
#         Computes the integral of a given scalar function `u` over the domain, using the precomputed vector and the expansion of the function

#         .. math::
#             \int_\Omega u \,d\Omega = \sum_{k=1}^{N_h} a^k_u \cdot L_k = \boldsymbol{a}_u^T\cdot \boldsymbol{L}

#         Parameters
#         ----------
#         u : `Function` (or `np.ndarray`)
#             Function belonging to the same functional space V (it must be a scalar!)

#         Returns
#         -------
#         value : float
#             Integral over the domain
#         """
        
#         assert self.V.num_sub_spaces == 0, "Integral is only defined for scalar functions"
#         _u = self.check_input(u)

#         value = _u.T @ self.rhs_vec
            
#         return value

#     def average(self, u: Function):
#         r""" 
#         Computes the integral average of a given **scalar** function `u` over the domain

#         .. math::
#             \langle u \rangle = \frac{1}{|\Omega|}\int_\Omega u \,d\Omega

#         Parameters
#         ----------
#         u : `Function` (or `np.ndarray`)
#             Function belonging to the same functional space V (it must be a scalar!)

#         Returns
#         -------
#         ave_value : float
#             Average over the domain
#         """

#         one_function = Function(self.V)
#         one_function.x.set(1.0)
#         domain_integral = self.integral(one_function)

#         value = self.integral(u) / domain_integral

#         return value
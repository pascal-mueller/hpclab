from __future__ import print_function, unicode_literals

import sys
import ipopt
import numpy as np
import matplotlib.pyplot as plt

class example(object):
    def __init__(self, N):
        self.N = N
        self.h = 1.0/(N+1)
        self.s = (N+2)*(N+2)

    # Implemented (19)
    # OBJECTIVE FUNCTION F
    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        N = self.N
        s = self.s
        h = self.h 

        F = 0.0
        
        y = x[:s]
        u = x[s:]
        
        print("s=",s)
        print("len(x)=", len(x))
        print("len(y)=", len(y))
        print("len(u)=",len(u))
        print("N=", N)

        # INNER
        for i in range(1,N+1): # rows
            for j in range(1,N+1): # cols
                x1 = h*i 
                x2 = h*j 
                y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
                print("total = ", (N+2)*i+j)
                y_ij = y[(N+2)*i+j]
                 
                F += (y_ij - y_d)**2
        
        F *= h

        # BOUNDARY
        # I think we have vanishing dirichlet BC 
        for i in [0,N+1]: # rows
            for j in [0,N+1]: # cols
                x1 = h*i 
                x2 = h*j  
                #u_ij = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
                u_ij = u[(N+2)*i+j]

                F += 0.01 * u_ij*u_ij
        
        return 0.5*h*F
        #return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    # grad(F)=[ d_x1 F(x), ..., f_xn F(x)]
    # Note: While we only have a two dimensional domain [unit square]
    # are interested in the gradient of each discretization point. Thus this
    # gradient will be a (N+2)*(N+2) dimensional vector!
    #
    # The first (N+2) derivatives are for y
    # The second (N+2) derivatives are for u
    #
    # dF(y,u)/dy_ij = h*h*( (y_ij - y_ij^d) * (1.0 - d*y_ij^(d-1)) + 0
    # dF(y,u)/du_ij = alpha*h*( (u_ij - u_ij^d) * (1.0 - d*u_ij^(d-1)) )
    #
    # grad(F) = [idF(y,u)/dy_ij, dF(y,u)/du_ij]

    #
    # Note that: u_ij^d=K => d = log(u_ij - K) whereas K is the RHS of (26)
    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        
        N = self.N
        s = self.s
        h = self.h 
        
        y = x[:s]
        u = x[s:]

        h = 1.0/(N+1)
        gradF_y = np.zeros(len(y))
        gradF_u = np.zeros(len(u))
        
        # INTERIOR Note: only depends on y
        for i in range(1,N+1):
            for j in range(1,N+1):
                x1 = h*i 
                x2 = h*j 
                y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
                
                y_ij = y[(N+2)*i+j]
                d = np.log(y_ij) / np.log(y_d)
                
                # dF/dy_ij
                gradF_y[(N+2)*i+j] = h*h*( (y_ij - y_ij**d)
                    * (1.0 - d*y_ij**(d-1)))

        # BOUNDARY Note: only depends on u
        for i in [0,N+1]:
            for j in [1,N+1]:
                x1 = h*i 
                x2 = h*j 
                
                u_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
                u_ij = u[(N+2)*i+j]
                d = np.log(u_ij) / np.log(u_d) # u_d = y_d
                # dF/dy_ij
                gradF_u[(N+2)*i+j] = 0.01*h*u_ij
       
                print("\n\n\n\n\nA")
                quit()
        
        foo = np.concatenate(gradF_y, gradF_u)
        print("\n\n\n\n\n")
        print(foo)
        quit()

        return np.concatenate(gradF_u, gradF_y)
        """return np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]),
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])"""
    
    # The constraint is given by the laplacian in eq. (1).
    # To solve it, we also discretize it using finite differences, resuling in
    # the stencil operator you can see in (20).
    #
    # Note that G^h(y), since it's the constraint for y.
    # Note that we need to treat it special on the boundaries and corner points!
    #
    # BUT: We can see in (2) that we have vanishing Dirichlet BC
    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        N = self.N
        h = self.h
        
        G = np.zeros(len(x)) # Vanishing Dirichlet BC implied

        # INTERIOR - We assume the origin to be in the top left corner
        for i in range(1,N+1):
            for j in range(1,N+1):
                y_middle = x[(N+2)*i+j]
                y_left  = x[(N+2)*i+j-1]
                y_right = x[(N+2)*i+j+1]
                y_up    = x[(N+1)*(i-1)+j]
                y_down  = x[(N+1)*(i+1)+j]
                
                x1 = h*i 
                x2 = h*j 
                y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
                
                y_ij = x[(N+2)*i+j]
                d = np.log(y_ij) / np.log(y_d) # u_d = y_d
                # TODO: Possible mistake
                h2d = -20*h*h
                G[(N+2)*i+j] += 4*y_middle - y_down - y_up - y_right - y_left
                - h2d
        
        # Upper inner boundaries
        i = 0
        for j in range(1,N+1):
            y_middle = x[(N+2)*i+j]
            y_left  = x[(N+2)*i+j-1]
            y_right = x[(N+2)*i+j+1]
            y_down  = x[(N+1)*(i+1)+j]
            
            x1 = h*i 
            x2 = h*j 
            y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
            
            y_ij = x[(N+2)*i+j]
            d = np.log(y_ij) / np.log(y_d) # u_d = y_d
            # TODO: Possible mistake
            h2d = -20*h*h
            G[(N+2)*i+j] += 4*y_middle - y_down - y_right - y_left - h2d

        # Lower inner boundaries
        i = N+2
        for j in range(1,N+1):
            y_middle = x[(N+2)*i+j]
            y_left  = x[(N+2)*i+j-1]
            y_right = x[(N+2)*i+j+1]
            y_up    = x[(N+1)*(i-1)+j]
            
            x1 = h*i 
            x2 = h*j 
            y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
            
            y_ij = x[(N+2)*i+j]
            d = np.log(y_ij) / np.log(y_d) # u_d = y_d
            # TODO: Possible mistake
            h2d = -20*h*h
            G[(N+2)*i+j] += 4*y_middle - y_up - y_right - y_left - h2d

        # Left inner boundaries
        j = 0
        for i in range(1,N+1):
            y_middle = x[(N+2)*i+j]
            y_left  = x[(N+2)*i+j-1]
            y_right = x[(N+2)*i+j+1]
            
            x1 = h*i 
            x2 = h*j 
            y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
            
            y_ij = x[(N+2)*i+j]
            d = np.log(y_ij) / np.log(y_d) # u_d = y_d
            # TODO: Possible mistake
            h2d = -20*h*h
            G[(N+2)*i+j] += 4*y_middle - y_right - y_left - h2d

        # Right inner boundaries
        j = N+2
        for i in range(1,N+1):
            y_middle = x[(N+2)*i+j]
            y_left  = x[(N+2)*i+j-1]
            y_right = x[(N+2)*i+j+1]
            
            x1 = h*i 
            x2 = h*j 
            y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
            
            y_ij = x[(N+2)*i+j]
            d = np.log(y_ij) / np.log(y_d) # u_d = y_d
            # TODO: Possible mistake
            h2d = -20*h*h
            G[(N+2)*i+j] += 4*y_middle - y_right - y_left - h2d

        # Left upper corner
        y_middle = x[(N+2)*i+j]
        y_right = x[(N+2)*i+j+1]
        y_down  = x[(N+1)*(i+1)+j]
        
        x1 = h*i 
        x2 = h*j 
        y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
        
        y_ij = x[(N+2)*i+j]
        d = np.log(y_ij) / np.log(y_d) # u_d = y_d
        # TODO: Possible mistake
        h2d = -20*h*h
        G[(N+2)*i+j] += 4*y_middle - y_down - y_right - h2d

        
        # Right upper corner
        y_middle = x[(N+2)*i+j]
        y_left  = x[(N+2)*i+j-1]
        y_down  = x[(N+1)*(i+1)+j]
        
        x1 = h*i 
        x2 = h*j 
        y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
        
        y_ij = x[(N+2)*i+j]
        d = np.log(y_ij) / np.log(y_d) # u_d = y_d
        # TODO: Possible mistake
        h2d = -20*h*h
        G[(N+2)*i+j] += 4*y_middle - y_down - y_left - h2d
         
        # Right lower corner
        y_middle = x[(N+2)*i+j]
        y_left  = x[(N+2)*i+j-1]
        y_up    = x[(N+1)*(i-1)+j]
        
        x1 = h*i 
        x2 = h*j 
        y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
        
        y_ij = x[(N+2)*i+j]
        d = np.log(y_ij) / np.log(y_d) # u_d = y_d
        # TODO: Possible mistake
        h2d = -20*h*h
        G[(N+2)*i+j] += 4*y_middle - y_up - y_left - h2d

        # Left lower corner
        y_middle = x[(N+2)*i+j]
        y_right = x[(N+2)*i+j+1]
        y_up    = x[(N+1)*(i-1)+j]
        
        x1 = h*i 
        x2 = h*j 
        y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
        
        y_ij = x[(N+2)*i+j]
        d = np.log(y_ij) / np.log(y_d) # u_d = y_d
        # TODO: Possible mistake
        h2d = -20*h*h
        G[(N+2)*i+j] += 4*y_middle - y_up - y_right - h2d


        return G
        #return np.array((np.prod(x), np.dot(x, x)))
    
    # This is the jacobian of the constraints! Not of F!
    # => Jacobian(G)
    #
    # The jacobian in general is given by
    # J = [ dG(y)/dy1, ..., dG(y)/dy_n]
    #
    # So again: While we have a 2D problem analytically we have a (N+2) dim.
    # discretized problem here. We evaluate the jacobian on every point.
    #
    # With d_ij = log(y_ij - K) whereas K being the RHS of (26)  we get
    # dG(y_ij)/dy_ij = 4.0 + h*h*( 1.0/( y_ij - K ) )
    #
    # Further note: Since we have vanishing Dirichlet BC on G, we also won't
    # see any change on the boundary of G i.e. dG(y)/dy_ij = 0 on the boundary!
    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        
        N = self.N
        s = self.s
        h = self.h
        
        y = x[:s]

        jacobianG_y = np.zeros(len(x)) # Implies vanishing Dirichlet BC
        
        # INTERIOR
        for i in range(0,N+2):
            for j in range(0,N+2):
                jacobianG_y = 4.0 

        return jacobianG_y
        #return np.concatenate((np.prod(x) / x, 2*x))
    
    # This is the hessian of the lagragian
    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #

        return np.nonzero(np.tril(np.ones((self.N+2, self.N+2))))
    
    # This is the hessian of the lagragian! L = J(y,u) + lambda^T*c(y,u)
    #
    # Note that we only use the LOWER TIRNGULAR part according to
    # hessianstructure()!
    #
    # Hessian of the Lagrangian: (see notebook example)
    #   \nabla^2_? * obj_factor * F(y,u) + lagrange*\nabla_?^2 G(y)
    #
    # Our lagrangian is
    #   L(y,u) = F(y,u) + lambda^T * G(y)
    # => \nabla^2 L(y,u) = \nabla^2 F(y,u) + \lambda^T\nabla G(y)
    #
    # Remember that we basically treat [y,u] as (N+2)*(N+2) separate variables.
    # So our hessian will be of form (N+2) x (N+2)
    

    # The first derivatives of F are:
    # dF(y,u)/dy_ij = h*h*( (y_ij - y_ij^d) * (1.0 - d*y_ij^(d-1)) + 0
    # dF(y,u)/du_ij = alpha*h*( (u_ij - u_ij^d) * (1.0 - d*u_ij^(d-1)) )
    #
    # Diagonal:
    # d^2F(y,u)/dy_ij^2 = h*h*( [1.0 - d*y_ij^(d-1)] * [1.0 - d*y_ij^(d-1)]
    #                          + [u_ij - u_ij^d] * []) 
    #
    # The first derivatives of G are:
    # dG(y_ij)/dy_ij = 4.0 + h*h*( 1.0/( y_ij - K ) )
    # dG(y_ij)/dy_ij = h*h*( -1.0/( y_ij - K )^2 )
    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #

        N = self.N
        s = self.s
        h = self.h
        
        y = x[:s]
        u = x[s:]

        hess = np.zeros((N+2)*(N+2))
        
        # INTERIOR
        for i in range(1,N+1):
            x1 = h*i 
            x2 = h*i 
            y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
            
            y_ii = y[(N+2)*i+i]
            d = np.log(y_ii) / np.log(y_d)
            # Add F
            hessF = -d*(d-1)*y_ii**(d-2) 
            hessF *= y_ii - y_d
            hessF += ( 1.0 - d*y_d ) * ( 1.0 - d*y_ii**(d-1) )
            hessF *= obj_factor*h*h

            # Add G
            hessG = 0.0
            
            # Write hessian matrix element
            hess[(N+2)*i+i] = hessF

        # Boundary
        for i in [0,N+1]:
            x1 = h*i 
            x2 = h*i 
            y_d = 3.0 + 5.0 * x1 * (x1-1.0) * x2 * (x2-1.0)
            
            u_ii = y[(N+2)*i+i]
            d = np.log(u_ii) / np.log(y_d)
            hess[(N+2)*i+i] = 0.01*h*h * ( 1.0 - d*u_ii**(d-1) )
            

        """
        H = obj_factor*np.array((
                (2*x[3], 0, 0, 0),
                (x[3],   0, 0, 0),
                (x[3],   0, 0, 0),
                (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        row, col = self.hessianstructure()

        return H[row, col]"""

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


#
# Define the problem
#

# Discretization parameter
N = 3
h = 1.0/(N+1)

print(f"Starting with N={N}, h={h}")

# Helper vars
ones = np.ones( (N+2)*(N+2) )
zeros = np.zeros( (N+2)*(N+2) )
no_boundary = ones*1e-20

# Initial guess
# x0 = [1.0, 5.0, 5.0, 1.0]
x0 = np.concatenate([ones, ones]) # [y(x),u(x)]

#
# Lower and upper bound on variables
#
# y(x) <= 3.5 on Omega
# 0 <= u(x) <= 10
lb = np.concatenate([no_boundary , no_boundary])
ub = np.concatenate([10.0*ones, no_boundary])

#
# Lower and upper bound on constraint G^h(y)
#
cl = np.concatenate([no_boundary, no_boundary])
cu = np.concatenate([no_boundary, no_boundary])


nlp = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=example(N),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )

#
# Set solver options
#
nlp.addOption('tol', 1e-7)

#
# Solve the problem
#
x, info = nlp.solve(x0)

print("Solution of the primal variables: x=%s\n" % repr(x))
print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
print("Objective=%s\n" % repr(info['obj_val']))

plt.plot(x)


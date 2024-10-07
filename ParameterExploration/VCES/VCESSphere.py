from collections import deque
from firedrake import *
try:
    import netgen
except ImportError:
    import sys
    warning("Unable to import NetGen.")
    sys.exit(0)

from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import copy
import csv
import time


# Set Current working directory
os.chdir("/home/stefano/Desktop/stefano-assist/ParameterExploration/VCES")

geo = SplineGeometry()
tolerance = 1e-10
max_iterations = 8
width = 2


parser = argparse.ArgumentParser(description='Set Mesh Refinement Parameters')
parser.add_argument('--thresh', type=float, nargs=2,
                    default=[0.2, 0.8], help='Brackets for marking elements')
args = parser.parse_args()

# Use command-line arguments
lbracket, ubracket = args.thresh
bracket = [lbracket, ubracket]


NumCells = []
# Generate initial mesh using netgen
TriHeight = .2
geo = SplineGeometry()
geo.AddRectangle(p1=(-1*width, -1*width),
                 p2=(1*width, 1*width),
                 bc="rectangle")

ngmsh = geo.GenerateMesh(maxh=TriHeight)
mesh = Mesh(ngmsh)
mesh.topology_dm.viewFromOptions('-dm_view')
meshHierarchy = [mesh]


def Mark(msh, u, lb, iter, bracket=[.4, .8]):
    W = FunctionSpace(msh, "DG", 0)

    # Heat Equation Scheme
    # ----------------------------------------------------------------------------------
    V = FunctionSpace(msh, "CG", 1)
    u_ = Function(V, name="CurrentStep")
    CGHeatEQ = Function(V, name="CGHeatEQ")
    v = TestFunction(V)
    DiffCG = Function(V, name="CG Indicator").interpolate(abs(u - lb))
    CGActiveIndicator = Function(V, name="OuterIntersectionCG").interpolate(
        conditional(DiffCG < tolerance, 0, 1))
    u_.assign(CGActiveIndicator)

    timestep = 1.0/int(msh.num_cells()**(.5))

    F = (inner((CGHeatEQ - u_)/timestep, v) +
         inner(grad(CGHeatEQ), grad(v))) * dx
    solve(F == 0, CGHeatEQ)
    # ----------------------------------------------------------------------------------
    # We send our one-step heat equation in to DG0 via interpolate to keep bounds (0, 1)
    DGHeatEQ = Function(W, name="DGHeatEQ").interpolate(CGHeatEQ)

    towrite = (u, CGActiveIndicator, CGHeatEQ, DGHeatEQ)
    File('output/AMRMarkingFunction: %s.pvd' % iter).write(*towrite)

    # Mark boundary elements
    mark = Function(W, name="Final Marking").interpolate(
        conditional(DGHeatEQ > bracket[0], conditional(DGHeatEQ < bracket[1], 1, 0), 0))

    towrite = (mark, u, CGActiveIndicator, CGHeatEQ, DGHeatEQ)
    File('output/AMRMarkingFunction: %s.pvd' % iter).write(*towrite)
    return mark


for i in range(max_iterations):
    NumCells.append(meshHierarchy[-1].num_cells())
    print("level {}".format(i))
    mesh = meshHierarchy[-1]
    # obstacle and solution are in P1
    V = FunctionSpace(mesh, "CG", 1)
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    # see Chapter 12 of Bueler (2021)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = - r0 / psi0
    psi_ufl = conditional(le(r, r0), sqrt(1.0 - r * r),
                          psi0 + dpsi0 * (r - r0))
    lb = interpolate(psi_ufl, V)
    # exact solution is known (and it determines Dirichlet boundary)
    afree = 0.697965148223374
    A = 0.680259411891719
    B = 0.471519893402112
    gbdry_ufl = conditional(le(r, afree), psi_ufl, - A * ln(r) + B)
    gbdry = interpolate(gbdry_ufl, V)
    uexact = gbdry.copy()

    # initial iterate is zero
    if i == 0:
        u = Function(V, name="u (FE soln)")
    else:
        # Need to define a destination function space to make cross mesh interpolation work
        V_dest = FunctionSpace(mesh, "CG", 1)
        u = interpolate(u, V_dest)

    # weak form problem; F is residual operator in nonlinear system F==0
    v = TestFunction(V)
    # as in Laplace equation:  - div (grad u) = 0
    F = inner(grad(u), grad(v)) * dx
    bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
    bcs = DirichletBC(V, gbdry, bdry_ids)

    # problem is nonlinear so we need a nonlinear solver, from PETSc's SNES component
    # specific solver is a VI-adapted line search Newton method called "vinewtonrsls"
    # see reference:
    #   S. Benson and T. Munson (2006). Flexible complementarity solvers for large-scale applications,
    #       Optimization Methods and Software, 21, 155–168.
    sp = {"snes_monitor": None,         # prints residual norms for each Newton iteration
          "snes_type": "vinewtonrsls",
          "snes_converged_reason": None,  # prints CONVERGED_... message at end of solve
          "snes_vi_monitor": None,       # prints bounds info for each Newton iteration
          "snes_rtol": 1.0e-8,
          "snes_atol": 1.0e-12,
          "snes_stol": 1.0e-12,
          "snes_vi_zero_tolerance": 1.0e-12,
          "snes_linesearch_type": "basic",
          # these 3 options say Newton step equations are solved by LU
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="")
    ub = interpolate(Constant(PETSc.INFINITY), V)  # no upper obstacle in fact
    # essentially same as:  solve(F == 0, u, bcs=bcs, ...
    solver.solve(bounds=(lb, ub))

    (mark) = Mark(mesh, u, lb, i, bracket)

    nextmesh = mesh.refine_marked_elements(mark)
    meshHierarchy.append(nextmesh)

output_file = 'VCESwith({}).csv'.format(bracket)

# Write the arrays to the CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['NumCells'])
    for data in zip(NumCells):
        writer.writerow(data)  # Write each row of data

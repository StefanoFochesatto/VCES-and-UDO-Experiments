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
os.chdir("/home/stefano/Desktop/stefano-assist/ParameterExploration/UDO")

geo = SplineGeometry()
tolerance = 1e-10
max_iterations = 8
width = 2

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Set Mesh Refinement Parameters')
parser.add_argument('--n', type=int, default=1,
                    help='neighborhood depth')
args = parser.parse_args()

# Use command-line arguments
neighbors = args.n


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


def dilation(mesh, func, func_space):
    result = Function(func_space, name='DilatedNeighbors')

    num_cells = mesh.num_cells()
    cell_closure = mesh.topology.cell_closure

    # Dictionary to map edges to the cells sharing each edge
    edge_to_cells = {}

    for cell_id in range(num_cells):
        edges = cell_closure[cell_id][3:6]
        for edge in edges:
            if edge not in edge_to_cells:
                edge_to_cells[edge] = []
            edge_to_cells[edge].append(cell_id)

    for edge, cells in edge_to_cells.items():
        # We consider edges that are in the interior of the domain.
        if len(cells) == 2:
            cell1, cell2 = cells
            if func.dat.data[cell1] != func.dat.data[cell2]:
                if func.dat.data[cell1] == 0:
                    result.dat.data[cell1] = 1
                else:
                    result.dat.data[cell2] = 1

    return result


# Slow MultiNeighbor Lookup
# def mark_neighbors(mesh, func):
#     # Create a new DG0 function to store the result
#     result = Function(FunctionSpace(mesh, "DG", 0))

#     # Get the cell closure
#     cell_vertex_map = mesh.topology.cell_closure

#     # Loop over all cells
#     for i in range(mesh.num_cells()):
#         # If the function value is 1
#         if func.dat.data[i] == 1:
#             # Get the vertices of the cell
#             vertices = cell_vertex_map[i]

#             # Loop over all cells again to find the neighbors
#             for j in range(mesh.num_cells()):
#                 # If the cell shares a vertex with the original cell, it's a neighbor
#                 if any(vertex in cell_vertex_map[j] for vertex in vertices):
#                     # Mark the neighbor
#                     result.dat.data[j] = 1

#     return result

# Slow Single Neighbor Lookup
# def one_neighbor(mesh, func, func_space, ActiveSet):
#     # Create a new DG0 function to store the result
#     result = Function(func_space)

#     # Get the cell closure
#     cell_vertex_map = mesh.topology.cell_closure

#     # Loop over all cells
#     for i in range(mesh.num_cells()):
#         # If the function value is 1
#         if func.dat.data[i] == 1:
#             # Mark the original cell
#             result.dat.data[i] = 1
#             # Get the vertices of the cell
#             vertices = cell_vertex_map[i]

#             # Loop over all cells again to find the neighbors
#             for j in range(mesh.num_cells()):
#                 # If the cell shares a vertex with the original cell and is in the ActiveSet
#                 if any(vertex in cell_vertex_map[j] for vertex in vertices) and ActiveSet.dat.data[j] == 1:
#                     # Mark the neighbor
#                     result.dat.data[j] = 1

#     return result

# Fast Single Neighbor Lookup


def one_neighbor(mesh, func, func_space, ActiveSet):
    # Create a new DG0 function to store the result
    result = Function(func_space)

    # Get the cell closure
    cell_vertex_map = mesh.topology.cell_closure

    # Create a dictionary to map vertices to the cells they belong to
    vertex_to_cells = {}
    for cell_id in range(mesh.num_cells()):
        for vertex in cell_vertex_map[cell_id]:
            if vertex not in vertex_to_cells:
                vertex_to_cells[vertex] = set()
            vertex_to_cells[vertex].add(cell_id)

    # Store cells to be activated in a set for efficient checks
    cells_to_activate = set()

    # Loop over all cells
    for i in range(mesh.num_cells()):
        # If the function value is 1
        if func.dat.data[i] == 1:
            # Mark the original cell
            result.dat.data[i] = 1
            cells_to_activate.add(i)

    # Activate neighbors
    for i in cells_to_activate:
        vertices = cell_vertex_map[i]
        for vertex in vertices:
            for neighbor in vertex_to_cells[vertex]:
                if ActiveSet.dat.data[neighbor] == 1:
                    result.dat.data[neighbor] = 1

    return result


# Fast Multi Neighbor Lookup BDF Avoids Active Set
def mark_neighbors(mesh, func, func_space, levels, ActiveSet):
    # Create a new DG0 function to store the result
    result = Function(func_space, name='nNeighbors')

    # Create a dictionary to map each vertex to the cells that contain it
    vertex_to_cells = {}

    # Get the cell to vertex connectivity
    cell_vertex_map = mesh.topology.cell_closure

    # Loop over all cells to populate the dictionary
    for i in range(mesh.num_cells()):
        # Only consider the first three entries, which correspond to the vertices
        for vertex in cell_vertex_map[i][:3]:
            if vertex not in vertex_to_cells:
                vertex_to_cells[vertex] = []
            vertex_to_cells[vertex].append(i)

    # Loop over all cells
    for i in range(mesh.num_cells()):
        # If the function value is 1 and the cell is in the active set
        if func.dat.data[i] == 1 and ActiveSet.dat.data[i] == 0:
            # Use a BFS algorithm to find all cells within the specified number of levels
            queue = deque([(i, 0)])
            visited = set()
            while queue:
                cell, level = queue.popleft()
                if cell not in visited and level <= levels:
                    visited.add(cell)
                    result.dat.data[cell] = 1
                    for vertex in cell_vertex_map[cell][:3]:
                        for neighbor in vertex_to_cells[vertex]:
                            # if ActiveSet.dat.data[neighbor] == 0:
                            queue.append((neighbor, level + 1))

    return result


def NodeMark(msh, u, lb, iter):
    # Compute pointwise indicator using CG 1
    U = FunctionSpace(msh, "CG", 1)
    W = FunctionSpace(msh, "DG", 0)

    psiCG = Function(U).interpolate(lb)
    DiffCG = Function(U, name="CG Indicator").interpolate(abs(u - psiCG))

    # Using conditional to get elementwise indictor for active set.
    # This piece of code does what we want, all nodes have to be less
    # than the tolerance for element to be active. (Checked by comparing pointwise CG 1 alternative.)
    ActiveSet = Function(W, name="Active Set").interpolate(
        conditional(DiffCG < tolerance, 1, 0))

    # Generating a CG 1 version of the active indicator
    ActiveSetCG = Function(U, name="OuterIntersectionCG").interpolate(
        conditional(DiffCG < tolerance, 1, 0))

    # Strict conditionals get us elements with non-zero gradient. These are the elements which
    # border the active set
    OuterElements = Function(W, name="OuterNeighborElements").interpolate(
        conditional(ActiveSetCG > 0, conditional(ActiveSetCG < 1, 1, 0), 0))

    # Returns OuterElements and n-neighbors. A cell is a neighbor if it contains a shared vertex. We also pass the active set
    # to ignore those elements during bfs
    # nNeighbors = one_neighbor(msh, OuterElements, W, ActiveSet)
    nNeighbors = mark_neighbors(msh, OuterElements, W, neighbors, ActiveSet)
    DilatedNeighbors = dilation(msh, nNeighbors, W)
    towrite = (DiffCG, ActiveSet, ActiveSetCG, OuterElements,
               nNeighbors, DilatedNeighbors, u)
    File('NodewiseSphere/MarkingFunction: %s.pvd' % iter).write(*towrite)

    return nNeighbors


if __name__ == "__main__":
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
        sp = {"snes_vi_monitor": None,         # prints residual norms for each Newton iteration
              "snes_type": "vinewtonrsls",
              "snes_converged_reason": None,  # prints CONVERGED_... message at end of solve
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
        ub = interpolate(Constant(PETSc.INFINITY), V)
        solver.solve(bounds=(lb, ub))

        (mark) = NodeMark(mesh, u, lb, i)

        nextmesh = mesh.refine_marked_elements(mark)
        meshHierarchy.append(nextmesh)

    output_file = 'UDOwith({}).csv'.format(neighbors)

    # Write the arrays to the CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['NumCells'])
        for data in zip(NumCells):
            writer.writerow(data)  # Write each row of data

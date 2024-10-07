from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely import hausdorff_distance

import math

from firedrake import *
try:
    import netgen
except ImportError:
    import sys
    warning("Unable to import NetGen.")
    sys.exit(0)
from netgen.geom2d import SplineGeometry

import csv


geo = SplineGeometry()
max_iterations = 8
width = 2
markbracket = [0.2, 0.8]
TriHeight = .45
tolerance = 1e-10


def create_circle(center, radius, num_points):

    # Generate the points on the circle
    points = [(center[0] + radius * math.cos(2 * math.pi * i / num_points),
               center[1] + radius * math.sin(2 * math.pi * i / num_points))
              for i in range(num_points)]

    return Polygon(points)


AnalyticFreeBoundary = create_circle((0, 0), 0.697965148223374, 7500)


def GetProblem(mesh, u, i):
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

    return (V, u, lb, bcs, F, uexact)


def GetSolver(F, u, lb, bcs, V):
    # problem is nonlinear so we need a nonlinear solver, from PETSc's SNES component
    # specific solver is a VI-adapted line search Newton method called "vinewtonrsls"
    # see reference:
    #   S. Benson and T. Munson (2006). Flexible complementarity solvers for large-scale applications,
    #       Optimization Methods and Software, 21, 155–168.
    sp = {"snes_monitor": None,         # prints residual norms for each Newton iteration
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
    # no upper obstacle in fact
    ub = interpolate(Constant(PETSc.INFINITY), V)
    # essentially same as:  solve(F == 0, u, bcs=bcs, ...
    solver.solve(bounds=(lb, ub))

    return (F, u, lb, bcs, sp, problem, solver, ub)


def Mark(msh, u, lb, iter, Refinement, bracket=[.4, .8]):
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

    # Mark boundary elements
    mark = Function(W, name="Final Marking").interpolate(
        conditional(DGHeatEQ > bracket[0], conditional(DGHeatEQ < bracket[1], 1, 0), 0))

    towrite = (mark, u, CGActiveIndicator, CGHeatEQ, DGHeatEQ)
    if Refinement == 'AMR':
        File('output/AMRMarkingFunction: %s.pvd' % iter).write(*towrite)
    elif Refinement == 'Unif':
        File('output/UNIFMarkingFunction: %s.pvd' % iter).write(*towrite)
    else:
        File('output/HybridMarkingFunction: %s.pvd' % iter).write(*towrite)

    return mark


def GetMeshDetails(mesh, name='', color=None):
    '''Print mesh information using DMPlex numbering.'''
    plex = mesh.topology_dm             # DMPlex
    coords = plex.getCoordinatesLocal()  # Vec
    vertices = plex.getDepthStratum(0)  # pair
    edges = plex.getDepthStratum(1)     # pair
    triangles = plex.getDepthStratum(2)  # pair
    ntriangles = triangles[1]-triangles[0]
    nvertices = vertices[1]-vertices[0]
    nedges = edges[1]-edges[0]
    print(color % '%s has %d elements, %d vertices, and %d edges:' %
          (name, ntriangles, nvertices, nedges))
    # print(color % '  coordinates of vertices (DMPlex numbering)')
    # for j in range(nvertices):
    #    print('    %2d: (%f,%f)' %
    #          (j + vertices[0], coords.array[2*j], coords.array[2*j+1]))
    # cells = mesh.cell_closure
    # print(color % '  cell closure discrete structure (DMPlex numbering) is %d x %d' %
    #      (np.shape(cells)[0], np.shape(cells)[1]))
    # print(cells)
    return (ntriangles, nvertices, nedges)


def get_active_vertices(ActiveSetCG, OuterElements, mesh):
    # Extract the function values
    active_set_v = ActiveSetCG.dat.data
    outer_elements_v = OuterElements.dat.data

    # Get the coordinates of the mesh vertices
    # coords = mesh.coordinates.dat.data
    coords = mesh.coordinates.dat.data_ro_with_halos
    cell_to_vertices = mesh.coordinates.cell_node_map().values_with_halo
    active_vertices = []
    active_vertices_idx = []

    # Iterate over the cells' closure
    for cell_index in range(mesh.topology.num_cells()):
        # Check if the current cell is marked by OuterElements
        if outer_elements_v[cell_index] == 1:
            cell_closure = cell_to_vertices[cell_index]
            for node_index in cell_closure:
                # Check if this vertex has a value of 1 in ActiveSetCG
                if ActiveSetCG.at(coords[node_index]) >= .5:
                    active_vertices.append(tuple(coords[node_index]))
                    active_vertices_idx.append(node_index)

    return active_vertices, active_vertices_idx


def sort_points_clockwise(points):
    # Compute the centroid of the points
    centroid = tuple(map(lambda x: sum(x) / len(points), zip(*points)))

    # Function to compute the angle relative to the centroid
    def angle_from_centroid(point):
        dx = point[0] - centroid[0]
        dy = point[1] - centroid[1]
        return math.atan2(dy, dx)

    # Sort the points based on the angle in descending order (clockwise)
    sorted_points = sorted(points, key=angle_from_centroid, reverse=True)
    # Append the first point to the end of the list to form a closed loop
    sorted_points.append(sorted_points[0])

    return sorted_points


def compute_iou(polygon1, polygon2):
    # Ensure the inputs are Shapely Polygons
    if not isinstance(polygon1, Polygon) or not isinstance(polygon2, Polygon):
        raise ValueError("Both inputs must be Shapely Polygon objects.")

    # Compute the intersection and union
    intersection = polygon1.intersection(polygon2)
    union = polygon1.union(polygon2)

    # Calculate the IoU
    intersection_area = intersection.area
    union_area = union.area
    iou = intersection_area / union_area

    return iou


def InitialMesh():
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1*width, -1*width),
                     p2=(1*width, 1*width),
                     bc="rectangle")

    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    mesh = Mesh(ngmsh)
    mesh.topology_dm.viewFromOptions('-dm_view')

    return (geo, mesh, ngmsh)


def HausdorffMetric(ComputedFreeBoundary, AnalyticFreeBoundary):
    # Compute the Hausdorff distance
    distance = hausdorff_distance(
        ComputedFreeBoundary, AnalyticFreeBoundary, .9)
    print("Hausdorff Distance: ", distance)
    return distance


def FindComputedFreeBoundary(mesh, u, iter):
    # Define psi obstacle function
    W = FunctionSpace(mesh, "DG", 0)
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = - r0 / psi0
    psi_ufl = conditional(le(r, r0), sqrt(1.0 - r * r),
                          psi0 + dpsi0 * (r - r0))

    # Compute pointwise Active indicator using CG 1
    U = FunctionSpace(mesh, "CG", 1)
    psiCG = Function(U).interpolate(psi_ufl)
    DiffCG = Function(U, name="CG Indicator").interpolate(u - psiCG)
    ActiveSet = Function(W, name="Active Set").interpolate(
        conditional(DiffCG < tolerance, 1, 0))

    # Indicator functions necessary(as of now) for searching for freeboundary vertices
    ActiveSetCG = Function(U, name="OuterIntersectionCG").interpolate(
        conditional(DiffCG < tolerance, 1, 0))
    OuterElements = Function(W, name="OuterNeighborElements").interpolate(
        conditional(ActiveSetCG > 0, conditional(ActiveSetCG < 1, 1, 0), 0))

    # Search for active vertices (slow)
    (ActiveVertices, ActiveVerticesIdx) = get_active_vertices(
        ActiveSetCG, OuterElements, mesh)

    # Sort vertices sow that we can form a polygon
    sorted = sort_points_clockwise(set(ActiveVertices))
    ComputedFreeBoundary = Polygon(sorted)

    # DebugCode
    # FreeboundaryIndicator = Function(U, name="FreeBoundaryIndicator")
    # FreeboundaryIndicator.dat.data[ActiveVerticesIdx] = 1
    # towrite = (DiffCG, ActiveSet, ActiveSetCG,
    #           OuterElements, FreeboundaryIndicator)
    # File('AreaMetric/Test: %s.pvd' % iter).write(*towrite)

    return ComputedFreeBoundary


def RunSolutionHybrid(max_iterations):
    l2 = []
    IoU = []
    Hausdorff = []
    count = []
    # Generate initial mesh using netgen
    (geo, mesh, ngmsh) = InitialMesh()
    meshHierarchy = [mesh]
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u (FE soln)")
    h = TriHeight

    for i in range(max_iterations):
        print("level {}: Hybrid".format(i))
        mesh = meshHierarchy[-1]
        # obstacle and solution are in P1
        (V, u, lb, bcs, F, uexact) = GetProblem(mesh, u, i)

        (F, u, lb, bcs, sp, problem, solver, ub) = GetSolver(F, u, lb, bcs, V)

        ComputedFreeBoundary = FindComputedFreeBoundary(mesh, u, i)
        IoUError = compute_iou(ComputedFreeBoundary, AnalyticFreeBoundary)
        IoU.append(IoUError)

        HausdorffError = HausdorffMetric(
            ComputedFreeBoundary, AnalyticFreeBoundary)
        Hausdorff.append(HausdorffError)

        diffu = interpolate(u - uexact, V)
        error_L2 = sqrt(assemble(dot(diffu, diffu) * dx))
        l2.append(error_L2)
        ratio = HausdorffError/(h**2)
        switch = math.isclose(ratio, 1, rel_tol=.1)

        if HausdorffError < h**2:
            mark = Mark(mesh, u, lb, i, 'Hybrid', bracket=[0, 1])
            nextmesh = mesh.refine_marked_elements(mark)
            meshHierarchy.append(nextmesh)
            h = h*(1/2)
            count.append(0)
        elif (switch):
            mark = Mark(mesh, u, lb, i, 'Hybrid', bracket=[0, 1])
            nextmesh = mesh.refine_marked_elements(mark)
            meshHierarchy.append(nextmesh)
            h = h*(1/2)
            count.append(0)
        else:
            mark = Mark(mesh, u, lb, i, 'Hybrid', markbracket)
            nextmesh = mesh.refine_marked_elements(mark)
            meshHierarchy.append(nextmesh)
            count.append(1)

        # if HausdorffError > h**2:
        #    mark = Mark(mesh, u, lb, i, 'AMR', markbracket)
        #    nextmesh = mesh.refine_marked_elements(mark)
        #    meshHierarchy.append(nextmesh)
        # else:
        #    mark = Mark(mesh, u, lb, i, 'Unif', bracket=[0, 1])
        #    nextmesh = mesh.refine_marked_elements(mark)
        #    meshHierarchy.append(nextmesh)
    for value in count:
        print(f"Unif is 0, AMR is 1:{value}")
    return (l2, IoU, Hausdorff, meshHierarchy)


if __name__ == "__main__":

    (l2, IoU, Hausdorff, meshHierarchy) = RunSolutionHybrid(max_iterations)
    Elements = []
    dof = []
    for mesh in meshHierarchy:
        (ntriangles, nvertices, nedges) = GetMeshDetails(
            mesh, name='AMR Mesh', color=BLUE)
        Elements.append(ntriangles)
        dof.append(nvertices)
    Elements.pop()
    dof.pop()

    output_file = 'Hybrid.csv'

    # Write the arrays to the CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['L2', 'IoU', 'Hausdorff', 'Elements', 'dof'])
        for data in zip(l2, IoU, Hausdorff, Elements, dof):
            writer.writerow(data)  # Write each row of data

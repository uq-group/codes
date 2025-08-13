from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.interpolate import LinearNDInterpolator


import numpy as np
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary
from mpi4py import MPI
from basix.ufl import element
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
import ufl
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector

from petsc4py import PETSc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pickle
from PIL import Image
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import zoom
import os

# Clean cache if compilation fails previously
try:
    for f in os.listdir("/root/.cache/fenics"):
        if f.startswith("libffcx_forms"):
            os.remove(f"/root/.cache/fenics/{f}")
except FileNotFoundError:
    pass
            
'''
Experiment
'''
# Simulation parameters
size_scale = 0.2
res = 200
AA = 5e-3
num_waves = 5
kk = np.pi * num_waves / size_scale
rho = 1.0
freq = 60.0
omega = 2 * np.pi * freq

# define number of samples
num_samples = 5
currentID = 0


# store all the samples
all_data = dict()
np.random.seed(53)

for id in range(currentID, currentID+num_samples):

    print('Generating {}-th sample'.format(id))

    # Create mesh
    msh = create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([size_scale, size_scale])], [res, res], CellType.triangle)
    vector_element = element("Lagrange", msh.basix_cell(), degree=1, shape=(2,))
    scalar_element = element("Lagrange", msh.basix_cell(), degree=1)
    V = functionspace(msh, vector_element)
    V_scalar = functionspace(msh, scalar_element)

    # Define boundary conditions
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    left_facets = locate_entities_boundary(msh, 1, left_boundary)
    left_dofs = locate_dofs_topological(V, 1, left_facets)
    def complex_disp(x):
        return np.vstack((np.zeros_like(x[1]), AA * np.exp(1j * kk * x[1])))
    disp_fun = Function(V, dtype=np.complex128)
    disp_fun.interpolate(complex_disp)
    bc_left = dirichletbc(disp_fun, left_dofs)
    bcs = [bc_left]

    class fun_gen():
        def __init__(self, image_file_path, mask_image_file_path, size_scale, mu_min=2.0, mu_max=4):
            """
            Load shear modulus and mask from image and initialize Gaussian Process for nu field.
            """
            self.size_scale = size_scale
            _, self.liver_mask_vals, tumor_mask_vals = self._load_mask_from_image(mask_image_file_path, 0.0, 1.0)
            self.E_coor, E_vals = self._load_E_from_image(image_file_path, self.liver_mask_vals, tumor_mask_vals, mu_min, mu_max)
            
            self.E_interp = LinearNDInterpolator(self.E_coor, E_vals)
            self.mask_interp = LinearNDInterpolator(self.E_coor, self.liver_mask_vals)

            kernel_nu = RBF(length_scale=0.4 * size_scale, length_scale_bounds=(1.0 * size_scale, 10.0 * size_scale))
            self.nu_gp = GaussianProcessRegressor(kernel=kernel_nu, alpha=1e-10)
        
        def _load_mask_from_image(self, image_path, mu_min, mu_max):
            """
            Convert an RGB image to scalar field using colormap inversion.
            """
            with open(image_path, 'rb') as f:
                data = pickle.load(f)
            w, h = data.shape
            liver_flag = np.zeros_like(data)
            tumor_flag = np.zeros_like(data)

            liver_flag[(data > 0)] = 1
            tumor_flag[data > 1] = 1

            # Create coordinate grid
            x = np.linspace(0, self.size_scale, w)
            y = np.linspace(0, self.size_scale, h)
            X, Y = np.meshgrid(x, y)
            coords = np.stack([X, Y], axis=-1).reshape(-1, 2)

            return coords, liver_flag.flatten(), tumor_flag.flatten()

        def _load_E_from_image(self, image_path, liver_mask_map, tumor_mask_map, mu_min, mu_max):
            """
            Convert an RGB image to scalar field using colormap inversion.
            """
            img = Image.open(image_path).convert("RGB")
            img_arr = np.array(img)
            h, w, _ = img_arr.shape
            cmap = cm.get_cmap("jet", 256)
            colormap_rgb = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            norm_vals = np.linspace(0, 12, 256)

            def closest_colormap_value(pixel_rgb):
                diffs = np.linalg.norm(colormap_rgb - pixel_rgb, axis=1)
                return norm_vals[np.argmin(diffs)]

            mu_map = np.zeros((h, w))
            for i in range(h):
                for j in range(w):
                    mu_map[i, j] = closest_colormap_value(img_arr[i, j])
            mu_map = mu_map.flatten()
            mu_map = mu_map * liver_mask_map
            mu_scaled = mu_min + (mu_map - mu_map.min()) / (mu_map.max() - mu_map.min()) * (mu_max - mu_min)
            mu_scaled[tumor_mask_map==1] = 5.0
            mu_scaled[self.liver_mask_vals==0] = 2.0

            # Create coordinate grid
            x = np.linspace(-0.005, self.size_scale+0.005, w)
            y = np.linspace(-0.005, self.size_scale+0.005, h)
            X, Y = np.meshgrid(x, y)
            coords = np.stack([X, Y], axis=-1).reshape(-1, 2)

            return coords, mu_scaled

        def is_point_inside_curve(self, xy):
            """
            Determine if points are inside the organ mask.
            """
            points = xy[:2, :].T
            mask_vals = self.mask_interp(points)
            return mask_vals > 0.5  # Threshold mask to binary

        def E_values(self, xy):
            """
            Return interpolated shear modulus at points.
            """
            points = xy[:2, :].T
            self.E_gp_samples = self.E_interp(points)
            random_ratio = np.random.rand() * 0.05 
            self.E_gp_samples = self.E_gp_samples + random_ratio * self.E_gp_samples * 1j
            # self.E_gp_samples = (self.E_gp_samples *0 +3) + np.zeros_like(self.E_gp_samples) * 1j

            return self.E_gp_samples
        
        def lambda_values(self, xy):
            """
            Return interpolated lambda values at points based on Gaussian Process for nu.
            """
        
            points = xy[:2, :].T
            gp_samples_for_fit = points[np.random.choice(points.shape[0], min(1000, len(points))), :]
            random_values = np.random.randn(len(gp_samples_for_fit)) * np.random.rand() * 0.1
            self.nu_gp.fit(gp_samples_for_fit, random_values)

            nu_gp_samples, _ = self.nu_gp.predict(points, return_std=True)
            nu_gp_samples[nu_gp_samples <= 0] = 0
            nu_gp_samples = (nu_gp_samples - nu_gp_samples.min()) / (nu_gp_samples.max() - nu_gp_samples.min()) * 0.05
            nu_gp_samples += 0.40
            inside = self.is_point_inside_curve(xy)
            nu_gp_samples[~inside] = 0.45

            E_vals = self.E_values(xy)
            lambda_gp_samples = self.E_gp_samples * 2 * nu_gp_samples / (1 - 2 * nu_gp_samples)

            return lambda_gp_samples


    # Example usage
    stage_id = np.random.randint(1,5)
    funcls = fun_gen(r'./Processed_data/color/field_{}.png'.format(id), r'./Processed_data/liver_masks/mask_{}.pkl'.format(id), size_scale)

    # Create the indicator function in the scalar function space
    indicator_function = Function(V_scalar)
    indicator_function.interpolate(lambda x: funcls.is_point_inside_curve(x))

    # Visualize geometry
    plt.figure(figsize=(3, 3))
    coordinates = msh.geometry.x
    grid_x, grid_y = np.mgrid[0:1*size_scale:res*1j, 0:1*size_scale:res*1j]
    E_img = scipy.interpolate.griddata(coordinates[:, :2], indicator_function.x.array.real, (grid_x, grid_y), method='linear', fill_value=0)
    sns.heatmap(E_img)
    plt.savefig('./test_geo.png')

    # Interpolate E and nu into scalar function spaces
    print('Generating the PDE parameters ...')
    E_function = Function(V_scalar, dtype=np.complex128)
    E_function.interpolate(lambda x: funcls.E_values(x))

    lambda_function = Function(V_scalar, dtype=np.complex128)
    lambda_function.interpolate(lambda x: funcls.lambda_values(x))

    '''
    compute PDE solutions
    '''
    print('Solving the displacements ...')
    # Define strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    def sigma(u):
        eps = epsilon(u)
        mu = ufl.variable(E_function)
        lam = ufl.variable(lambda_function)
        return 2 * mu * eps + lam * ufl.tr(eps) * ufl.Identity(2)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = Constant(msh, PETSc.ScalarType((0.0 + 0.0j, 0.0 + 0.0j))) # Body force
    a = form(ufl.inner(sigma(u), epsilon(v)) * ufl.dx - rho * omega**2 * ufl.inner(u, v) * ufl.dx)
    L = form(ufl.dot(f, ufl.conj(v)) * ufl.dx)
    # Assemble the system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve the problem
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setFromOptions()
    x = create_vector(a)
    solver.solve(b, x)

    # Extract the displacement
    displacements = x.array.reshape((-1, 2))
    coordinates = msh.geometry.x
    # Use mesh vertex coordinates (shape: (N, 3))
    coords = msh.geometry.x  # shape (3, N)
    # Evaluate functions at these physical coordinates
    mu_vals = E_function.x.array # .eval(coords, msh)[0]
    lam_vals = lambda_function.x.array

    # Grid interpolation for image representation
    grid_x, grid_y = np.mgrid[0:1*size_scale:res*1j, 0:1*size_scale:res*1j]
    u_real = scipy.interpolate.griddata(coordinates[:,:2], displacements[:, 0].real, (grid_x, grid_y), method='linear', fill_value=0)
    v_real = scipy.interpolate.griddata(coordinates[:,:2], displacements[:, 1].real, (grid_x, grid_y), method='linear', fill_value=0)
    u_imag = scipy.interpolate.griddata(coordinates[:,:2], displacements[:, 0].imag, (grid_x, grid_y), method='linear', fill_value=0)
    v_imag = scipy.interpolate.griddata(coordinates[:,:2], displacements[:, 1].imag, (grid_x, grid_y), method='linear', fill_value=0)
    mu_real = scipy.interpolate.griddata(coordinates[:,:2], mu_vals.real, (grid_x, grid_y), method='linear', fill_value=0)
    lam_real = scipy.interpolate.griddata(coordinates[:,:2], lam_vals.real, (grid_x, grid_y), method='linear', fill_value=0)
    mu_imag = scipy.interpolate.griddata(coordinates[:,:2], mu_vals.imag, (grid_x, grid_y), method='linear', fill_value=0)
    lam_imag = scipy.interpolate.griddata(coordinates[:,:2], lam_vals.imag, (grid_x, grid_y), method='linear', fill_value=0)

    # combine two images
    disp_map = np.concatenate(
        (np.expand_dims(u_real, 0), np.expand_dims(v_real, 0), np.expand_dims(u_imag, 0), np.expand_dims(v_imag, 0)), 0)
    stiff_map = np.concatenate(
        (np.expand_dims(mu_real, 0), np.expand_dims(lam_real, 0), np.expand_dims(mu_imag, 0), np.expand_dims(lam_imag, 0)), 0)

    # Save the data
    data = dict()
    data['xcoor'] = grid_x
    data['ycoor'] = grid_y
    data['U'] = disp_map
    data['mask'] = E_img
    data['mu'] = stiff_map
    data['omega_over_c'] = rho * (omega**2)

    # store it in the dataset
    all_data[id] = data

with open('data_general.pkl', 'wb') as handle:
    pickle.dump(all_data, handle)
print('All samples generated and saved to data_general.pkl')


import numpy as np
from tqdm import tqdm
import psutil

import CGNS

from filepath import *
from flow_properties import *

# ====================================================================================================
# ====================================================================================================

# Abertura do arquivo da malha
grid_file = grid_path + grid_name + grid_extension
ifile, nbases = CGNS.open_file_read(grid_file)
ibase = 1

# Leitura da malha O
izone = 1
idim_o = CGNS.zonedim_read(ifile, ibase, izone)
isize_o, nx_o, ny_o, nz_o = CGNS.zone_size_read(ifile, ibase, izone, idim_o)

# Dimensoes da malha O
ijk_min_o = [1, 1]
ijk_max_o = [nx_o, ny_o]

# Carregamento da malha O
xo = CGNS.read_2D_coord("CoordinateX", ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
yo = CGNS.read_2D_coord("CoordinateY", ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)

# Leitura da malha H
izone = 2
idim_h = CGNS.zonedim_read(ifile, ibase, izone)
isize_h, nx_h, ny_h, nz_h = CGNS.zone_size_read(ifile, ibase, izone, idim_h)

# Dimensoes da malha O
ijk_min_h = [1, 1]
ijk_max_h = [nx_h, ny_h]

# Carregamento da malha O
xh = CGNS.read_2D_coord("CoordinateX", ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
yh = CGNS.read_2D_coord("CoordinateY", ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)

# Cria um arquivo CGNS para armazeanr a media
mean_file_CGSN = mean_path_CGNS + mean_name + mean_extenrion
CGNS.create_file_cgns(mean_file_CGSN, '2D')

# Escrita das malhas no arquivo CGNS
CGNS.write_2D_coord(mean_file_CGSN, 1, nx_o, ny_o, xo, yo)
CGNS.write_2D_coord(mean_file_CGSN, 2, nx_h, ny_h, xh, yh)

# Escrita das malhas no arquivo Python
mean_file_python = mean_path_python + 'o_grid.npz'
np.savez(mean_file_python, x=xo, y=yo)

mean_file_python = mean_path_python + 'h_grid.npz'
np.savez(mean_file_python, x=xh, y=yh)

# 2D flow mean arrays
mean_density_o = np.zeros((nx_o, ny_o))
mean_density_h = np.zeros((nx_h, ny_h))

mean_pressure_o = np.zeros((nx_o, ny_o))
mean_pressure_h = np.zeros((nx_h, ny_h))

mean_temperature_o = np.zeros((nx_o, ny_o))
mean_temperature_h = np.zeros((nx_h, ny_h))

mean_momentumx_o = np.zeros((nx_o, ny_o))
mean_momentumy_o = np.zeros((nx_o, ny_o))
mean_momentumx_h = np.zeros((nx_h, ny_h))
mean_momentumy_h = np.zeros((nx_h, ny_h))

mean_velocityx_o = np.zeros((nx_o, ny_o))
mean_velocityy_o = np.zeros((nx_o, ny_o))
mean_velocityx_h = np.zeros((nx_h, ny_h))
mean_velocityy_h = np.zeros((nx_h, ny_h))

# Iblank
iblank_o = np.zeros((nx_o, ny_o))
iblank_h = np.zeros((nx_h, ny_h))

# Number and list of CGNS files
nqout = int((last_qout - first_qout) / skip_step_qout) + 1
qouts = range(first_qout, last_qout + 1, skip_step_qout)

# Read the current process
process = psutil.Process()

# Generate progress bar
with tqdm(total=nqout, desc="Processing qouts") as pbar:

    # Loop over CGNS files
    for qout in qouts:

        # Adjust the qout number to 6 numbers and convert to string
        nqout_str = str(qout).zfill(6)
        file_path_name = qout_path + qout_name + nqout_str + qout_extension

        # Open CGNS file
        ifile, nbases = CGNS.open_file_read(file_path_name)
        ibase = 1

        # Leitura da malha O
        izone = 1

        # Load Iblank
        if qout == first_qout:
            iblank_o = CGNS.read_2D_flow('Iblank', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)

        density_o = CGNS.read_2D_flow('Density', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
        mean_density_o += density_o

        pressure_o = CGNS.read_2D_flow('Pressure', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
        mean_pressure_o += pressure_o

        mean_temperature_o += pressure_o / (density_o * R)

        momentumx_o = CGNS.read_2D_flow('MomentumX', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
        momentumy_o = CGNS.read_2D_flow('MomentumY', ifile, ibase, izone, ijk_min_o, ijk_max_o, nx_o, ny_o)
        mean_momentumx_o[:,:] += momentumx_o
        mean_momentumy_o[:,:] += momentumy_o

        mean_velocityx_o[:,:] += momentumx_o / density_o
        mean_velocityy_o[:,:] += momentumy_o / density_o

        # Leitura da malha 2
        izone = 2

        # Load Iblank
        if qout == first_qout:
            iblank_h = CGNS.read_2D_flow('Iblank', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)

        density_h = CGNS.read_2D_flow('Density', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
        mean_density_h += density_h

        pressure_h = CGNS.read_2D_flow('Pressure', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
        mean_pressure_h += pressure_h

        mean_temperature_h += pressure_h / (density_h * R)

        momentumx_h = CGNS.read_2D_flow('MomentumX', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
        momentumy_h = CGNS.read_2D_flow('MomentumY', ifile, ibase, izone, ijk_min_h, ijk_max_h, nx_h, ny_h)
        mean_momentumx_h = momentumx_h
        mean_momentumy_h = momentumy_h

        mean_velocityx_h[:,:] += momentumx_h / density_h
        mean_velocityy_h[:,:] += momentumy_h / density_h

        # Close CGNS file
        CGSN.close_file(ifile)

        # Get memory usage
        mem_usage = process.memory_info().rss / (1024 ** 2)

        # Update progress bar
        pbar.set_postfix({"Qout": nqout_str, "RAM MB": f"{mem_usage:.1f}"})
        pbar.update(1)


# Compute flow properties mean
mean_density_o /= nqout
mean_density_h /= nqout

mean_pressure_o /= nqout
mean_pressure_h /= nqout

mean_temperature_o /= nqout
mean_temperature_h /= nqout

mean_momentumx_o /= nqout
mean_momentumy_o /= nqout
mean_momentumx_h /= nqout
mean_momentumy_h /= nqout

mean_velocityx_o /= nqout
mean_velocityy_o /= nqout
mean_velocityx_h /= nqout
mean_velocityy_h /= nqout

# Write mean in CGNS mean file
CGNS.write_soln_2D(mean_file_CGSN, 1, nx_o, ny_o, iblank_o, 'Iblank')
CGNS.write_soln_2D(mean_file_CGSN, 2, nx_h, ny_h, iblank_h, 'Iblank')

CGNS.write_soln_2D(mean_file_CGSN, 1, nx_o, ny_o, mean_density_o, 'Density')
CGNS.write_soln_2D(mean_file_CGSN, 2, nx_h, ny_h, mean_density_h, 'Density')

CGNS.write_soln_2D(mean_file_CGSN, 1, nx_o, ny_o, mean_pressure_o, 'Pressure')
CGNS.write_soln_2D(mean_file_CGSN, 2, nx_h, ny_h, mean_pressure_h, 'Pressure')

CGNS.write_soln_2D(mean_file_CGSN, 1, nx_o, ny_o, mean_temperature_o, 'Temperature')
CGNS.write_soln_2D(mean_file_CGSN, 2, nx_h, ny_h, mean_temperature_h, 'Temperature')

CGNS.write_soln_2D(mean_file_CGSN, 1, nx_o, ny_o, mean_momentumx_o, 'MomentumX')
CGNS.write_soln_2D(mean_file_CGSN, 2, nx_h, ny_h, mean_momentumx_h, 'MomentumX')
CGNS.write_soln_2D(mean_file_CGSN, 1, nx_o, ny_o, mean_momentumy_o, 'MomentumY')
CGNS.write_soln_2D(mean_file_CGSN, 2, nx_h, ny_h, mean_momentumy_h, 'MomentumY')

CGNS.write_soln_2D(mean_file_CGSN, 1, nx_o, ny_o, mean_velocityx_o, 'VelocityX')
CGNS.write_soln_2D(mean_file_CGSN, 2, nx_h, ny_h, mean_velocityx_h, 'VelocityX')
CGNS.write_soln_2D(mean_file_CGSN, 1, nx_o, ny_o, mean_velocityy_o, 'VelocityY')
CGNS.write_soln_2D(mean_file_CGSN, 2, nx_h, ny_h, mean_velocityy_h, 'VelocityY')

# Escrita das malhas no arquivo Python
mean_file_python = mean_path_python + 'iblank_o.npy'
np.save(mean_file_python, iblank_o)
mean_file_python = mean_path_python + 'iblank_h.npy'
np.save(mean_file_python, iblank_h)

mean_file_python = mean_path_python + 'mean_density_o.npy'
np.save(mean_file_python, mean_density_o)
mean_file_python = mean_path_python + 'mean_density_h.npy'
np.save(mean_file_python, mean_density_h)

mean_file_python = mean_path_python + 'mean_pressure_o.npy'
np.save(mean_file_python, mean_pressure_o)
mean_file_python = mean_path_python + 'mean_pressure_h.npy'
np.save(mean_file_python, mean_pressure_h)

mean_file_python = mean_path_python + 'mean_temperature_o.npy'
np.save(mean_file_python, mean_temperature_o)
mean_file_python = mean_path_python + 'mean_temperature_h.npy'
np.save(mean_file_python, mean_temperature_h)

mean_file_python = mean_path_python + 'mean_momentumx_o.npy'
np.save(mean_file_python, mean_momentumx_o)
mean_file_python = mean_path_python + 'mean_momentumx_h.npy'
np.save(mean_file_python, mean_momentumx_h)
mean_file_python = mean_path_python + 'mean_momentumy_o.npy'
np.save(mean_file_python, mean_momentumy_o)
mean_file_python = mean_path_python + 'mean_momentumy_h.npy'
np.save(mean_file_python, mean_momentumy_h)

mean_file_python = mean_path_python + 'mean_velocityx_o.npy'
np.save(mean_file_python, mean_velocityx_o)
mean_file_python = mean_path_python + 'mean_velocityx_h.npy'
np.save(mean_file_python, mean_velocityx_h)
mean_file_python = mean_path_python + 'mean_velocityy_o.npy'
np.save(mean_file_python, mean_velocityy_o)
mean_file_python = mean_path_python + 'mean_velocityy_h.npy'
np.save(mean_file_python, mean_velocityy_h)

# =========================================================================
#   Code with all the paths and file names to be read, load, or save.
#   New CGNS files, figures and data can be found in the path below.
#   Auxiliary information about flow properties analyses and to be done
# =========================================================================

# =====================================
#   Grid file
# =====================================
grid_path = '/media/lczoca/hd_leonardo2/Hugo_Data/'
grid_name = 'grid_2D'
grid_extension = '.cgns'

# =====================================
#   Solution files
# =====================================
qout_path = '/media/lczoca/hd_leonardo2/Hugo_Data/'
qout_name = 'qout'
qout_extension = '_2D.cgns'
first_qout     = 6284
last_qout      = 43809
skip_step_qout = 5

# =====================================
#   Mean solution file
# =====================================
mean_path = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/'
mean_path_CGNS = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/CGNS_data/'
mean_path_python = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/Python_data/'
mean_name = 'mean_2D_' + str(first_qout) + '_' + str(last_qout) + '_' + str(skip_step_qout)
mean_extenrion = '.cgns'

# =====================================
#   RMS solution file
# =====================================
rms_path = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/'
rms_path_CGNS = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/CGNS_data/'
rms_path_python = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/Python_data/'
rms_name = 'rms_2D_' + str(first_qout) + '_' + str(last_qout) + '_' + str(skip_step_qout)
rms_extenrion = '.cgns'

# =====================================
#   Flow solution file
# =====================================
flow_path = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/'
flow_path_python = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/Python_data2/'

# =====================================
#   Figures path
# =====================================
figure_path = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/Figure/'

# =========================================================================
# =========================================================================

# =====================================
#   Flow properties
# =====================================
gamma = 1.31                            # Ratio of specific heats
Ma = 2.0                                # Mach number
Re = 395000                             # Reynolds number (based on the freestream velocity )
Pr = 0.07182                            # Prandtl number (Ratio of the Sutherland constant over free-stream temperature)
R = (gamma - 1.0) / gamma               # Gas constant

# =====================================
#   Farfield properties
# =====================================
rho_infty = 1
P_infty = 1 / gamma
T_infty = 1 / (gamma - 1)
u_infty = 2
c_infty = (gamma * P_infty / rho_infty)**0.5

# =====================================
#   Read/load Region
# =====================================

# First point to read in coordinates (x, y)
nx_min = 1
ny_min = 1

# Last point to read in coordinates (x, y)
nx_max = 1280
ny_max = 300

# =====================================
#   Bubble properties
# =====================================

# Suction Side (SS)
bubble_length = 0.09823154306062291
separation_point = 0.7503580560086193
reattachment_point = 0.8453109140914615

# Pressure Side (PS)
# bubble_length =
# separation_point =
# reattachment_point =

# =====================================
#   Region to analyze
# =====================================

# Suction Side (SS)
# region_analysis = slice(227, 442)       # Bubble 0.4 < x < 0.92

# Pressure Side (PS)
region_analysis = slice(850, 1200)      # Bubble ... 


# =====================================
#   Probes to analyze
# =====================================

# Suction Side (SS)
# probe = [[50 , 0], 
#          [100, 0], 
#          [150, 0],
#          [200, 0],
#          [250, 0], 
#          [300, 0],
#          [350, 0],
#          [375, 0],
#          [400, 0],
#          [450, 0],
#          [500, 0]]

# probe = [[50 , 50], 
#          [100, 50], 
#          [150, 50],
#          [200, 50],
#          [250, 50], 
#          [300, 50],
#          [350, 50],
#          [375, 50],
#          [400, 50],
#          [450, 50],
#          [500, 50]]

# probe = [[50 , 100], 
#          [100, 100], 
#          [150, 100],
#          [200, 100],
#          [250, 100], 
#          [300, 100],
#          [350, 100],
#          [375, 100],
#          [400, 100],
#          [450, 100],
#          [500, 100]]

# probe = [[50 , 150], 
#          [100, 150], 
#          [150, 150],
#          [200, 150],
#          [250, 150], 
#          [300, 150],
#          [350, 150],
#          [375, 150],
#          [400, 150],
#          [450, 150],
#          [500, 150]]

# probe = [[50 , 200], 
#          [100, 200], 
#          [150, 200],
#          [200, 200],
#          [250, 200], 
#          [300, 200],
#          [350, 200],
#          [375, 200],
#          [400, 200],
#          [450, 200],
#          [500, 200]]

# Pressure Side (PS)
# probe = [[750 , 0], 
#          [800 , 0],
#          [850 , 0],
#          [900 , 0], 
#          [950 , 0],
#          [975 , 0],
#          [1000, 0],
#          [1050, 0],
#          [1100, 0],
#          [1150, 0],
#          [1200, 0]]

# probe = [[750 , 50], 
#          [800 , 50],
#          [850 , 50],
#          [900 , 50], 
#          [950 , 50],
#          [975 , 50],
#          [1000, 50],
#          [1050, 50],
#          [1100, 50],
#          [1150, 50],
#          [1200, 50]]

# probe = [[750 , 100], 
#          [800 , 100],
#          [850 , 100],
#          [900 , 100], 
#          [950 , 100],
#          [975 , 100],
#          [1000, 100],
#          [1050, 100],
#          [1100, 100],
#          [1150, 100],
#          [1200, 100]]

# probe = [[750 , 150], 
#          [800 , 150],
#          [850 , 150],
#          [900 , 150], 
#          [950 , 150],
#          [975 , 150],
#          [1000, 150],
#          [1050, 150],
#          [1100, 150],
#          [1150, 150],
#          [1200, 150]]

# probe = [[750 , 150], 
#          [800 , 150],
#          [850 , 150],
#          [900 , 150], 
#          [950 , 150],
#          [975 , 150],
#          [1000, 150],
#          [1050, 150],
#          [1100, 150],
#          [1150, 150],
#          [1200, 150]]

# probe = [[750 , 200], 
#          [800 , 200],
#          [850 , 200],
#          [900 , 200], 
#          [950 , 200],
#          [975 , 200],
#          [1000, 200],
#          [1050, 200],
#          [1100, 200],
#          [1150, 200],
#          [1200, 200]]

probe = [[750 , 0], 
         [800 , 0],
         [850 , 0],
         [900 , 0], 
         [950 , 0],
         [975 , 0],
         [1000, 0],
         [1050, 0],
         [1100, 0],
         [1150, 0],
         [1200, 0],
         [750 , 50], 
         [800 , 50],
         [850 , 50],
         [900 , 50], 
         [950 , 50],
         [975 , 50],
         [1000, 50],
         [1050, 50],
         [1100, 50],
         [1150, 50],
         [1200, 50],
         [750 , 100], 
         [800 , 100],
         [850 , 100],
         [900 , 100], 
         [950 , 100],
         [975 , 100],
         [1000, 100],
         [1050, 100],
         [1100, 100],
         [1150, 100],
         [1200, 100],
         [750 , 150], 
         [800 , 150],
         [850 , 150],
         [900 , 150], 
         [950 , 150],
         [975 , 150],
         [1000, 150],
         [1050, 150],
         [1100, 150],
         [1150, 150],
         [1200, 150],
         [750 , 150], 
         [800 , 150],
         [850 , 150],
         [900 , 150], 
         [950 , 150],
         [975 , 150],
         [1000, 150],
         [1050, 150],
         [1100, 150],
         [1150, 150],
         [1200, 150],
         [750 , 200], 
         [800 , 200],
         [850 , 200],
         [900 , 200], 
         [950 , 200],
         [975 , 200],
         [1000, 200],
         [1050, 200],
         [1100, 200],
         [1150, 200],
         [1200, 200]]


# Number of probes
num_probes = len(probe)

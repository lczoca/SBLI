# =========================================================================
#   Code with all the paths and file names to be read, load, or save.
#   New CGNS files, figures and data can be found in the path below.
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
flow_path_python = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/Python_data/'
# flow_name = 'flow_2D_' + str(first_qout) + '_' + str(last_qout) + '_' + str(skip_step_qout)
# flow_extenrion = '.cgns'

# =====================================
#   Figures path
# =====================================
figure_path = '/media/lczoca/hd_leonardo2/Hugo_Data_Processed/Figure/'

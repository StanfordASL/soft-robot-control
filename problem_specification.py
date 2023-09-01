### DIAMOND ROBOT PROBLEM SPECIFICATION ###
# Description
# from examples.diamond import diamond
# problem = diamond.collect_POD_data
# problem = diamond.collect_TPWL_data
# problem = diamond.run_scp

# JOHN: Run decaying trajectories and visualize
# various mode shapes
# problem = diamond.apply_constant_input

# from examples.diamond import diamond_rompc
# problem = diamond_rompc.run_rompc

# from examples.diamond import diamond_koopman
# problem = diamond_koopman.run_koopman

# from examples.hardware import calibration
# problem = calibration.output_node_calibration
# problem = calibration.rest_calibration
# problem = calibration.model_calibration
# problem = calibration.actuator_calibration

# from examples.hardware import diamond
# problem = diamond.run_scp
# problem = diamond.collect_POD_data
# problem = diamond.collect_TPWL_data
#problem = diamond.run_ilqr
#problem = diamond.TPWL_rollout
# problem = diamond.run_scp_OL

# from examples.hardware import diamond_rompc
# problem = diamond_rompc.run_rompc

# from examples.hardware import diamond_koopman
# problem = diamond_koopman.run_koopman
# problem = diamond_koopman.run_MPC_OL

# from examples.hardware import diamond_SSM
# problem = diamond_SSM.run_scp
# problem = diamond_SSM.run_scp_OL
# problem = diamond_SSM.module_test
# problem = diamond_SSM.module_test_continuous

# @ Jonas
# =============== TRUNK ROBOT =============== #

# from examples.trunk import calibration
# problem = calibration.rest_calibration
# problem = trunk.apply_constant_input

# from examples.trunk import trunk
# problem = trunk.sim_OL

# KOOPMAN
# from examples.trunk import trunk_koopman
# problem = trunk_koopman.run_koopman

# SSM
# from examples.trunk import trunk_SSM
# problem = trunk_SSM.run_scp

# ADIABATIC SSM
from examples.trunk import trunk_adiabaticSSM
problem = trunk_adiabaticSSM.run_scp

# TPWL
# from examples.trunk import trunk_tpwl
# # problem = trunk_tpwl.collect_POD_data
# # problem = trunk_tpwl.collect_TPWL_data
# problem = trunk_tpwl.run_scp
# # problem = trunk_tpwl.run_ilqr

from examples.hardware import diamond_SSM
import launch_sofa

problem = diamond_SSM.run_gusto_solver()
launch_sofa.main(problem)
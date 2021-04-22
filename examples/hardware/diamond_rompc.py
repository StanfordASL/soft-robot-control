import sys
from os.path import dirname, abspath, join

import numpy as np

path = dirname(abspath(__file__))
root = dirname(dirname(path))
sys.path.append(root)

from examples import Problem

# Default nodes are the "end effector (1354)" and the "elbows (726, 139, 1445, 729)"
DEFAULT_OUTPUT_NODES = [1354, 726, 139, 1445, 729]


def generate_linearized_ROM():
    """
    Generate a linear ROM by converting a TPWL model

    python3 diamond_rompc.py generate_linearized_rom
    """
    from sofacontrol.baselines.rompc.rompc_utils import TPWL2LinearROM

    tpwl_file = 'tpwl_model_snapshots'
    linrom_file = 'rompc_model'

    tpwl_loc = join(path, '{}.pkl'.format(tpwl_file))
    save_loc = join(path, '{}.pkl'.format(linrom_file))
    TPWL2LinearROM(tpwl_loc, save_loc)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'generate_linearized_rom':
        generate_linearized_ROM()
    else:
        raise RuntimeError('Not a valid command line argument')

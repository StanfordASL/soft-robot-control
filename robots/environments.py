import copy
import os
from math import cos
from math import sin
import sys
from pathlib import Path
import numpy as np

import Sofa.Core
from splib3.numerics import Quat, Vec3

from sofacontrol import measurement_models
from sofacontrol.utils import load_data

path = os.path.dirname(os.path.abspath(__file__))


class TemplateEnvironment:
    def __init__(self, name='Template', rayleighMass=0.1, rayleighStiffness=0.1, dt=0.01):
        self.name = name
        self.robot = Sofa.Core.Node(name)
        # set-up solvers
        self.robot.addObject('EulerImplicitSolver', name='odesolver', firstOrder="0", rayleighMass=str(rayleighMass),
                             rayleighStiffness=str(rayleighStiffness))
        self.robot.addObject('SparseLDLSolver', name='preconditioner', template="CompressedRowSparseMatrixd")
        self.robot.addObject('GenericConstraintCorrection', solverName="preconditioner")
        self.actuator_list = []
        self.nb_nodes = None
        self.gravity = [0., -9810., 0.]  # default
        self.dt = dt

    def get_measurement_model(self, nodes=None, pos=True, vel=True, qv=False):
        if nodes is None:
            return measurement_models.linearModel(range(self.nb_nodes), self.nb_nodes, pos=pos, vel=vel, qv=qv)
        else:
            return measurement_models.linearModel(nodes, self.nb_nodes, pos=pos, vel=vel, qv=qv)


class Trunk(TemplateEnvironment):
    def __init__(self, name='Trunk', youngModulus=450, poissonRatio=0.45, totalMass=0.042, inverseMode=False, all_cables=True, dt=0.01):
        super(Trunk, self).__init__(name=name, dt=dt)

        self.nb_nodes = 709
        self.gravity = [0., 0., 9810.]
        
        self.inverseMode = inverseMode
        self.robot.addObject('MeshVTKLoader', name='loader', filename=path + '/mesh/trunk.vtk')
        self.robot.addObject('MeshTopology', src='@loader', name='container')

        self.robot.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false',
                             showIndicesScale='4e-5')
        self.robot.addObject('UniformMass', totalMass=totalMass)
        self.robot.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large',
                             poissonRatio=poissonRatio, youngModulus=youngModulus)

        # Export every 10th step e.g., if dt = 0.01, save every 0.1 sec (defaults saving to disabled)
        self.robot.addObject('GlobalSystemMatrixExporter', exportEveryNumberOfSteps='10', enable='False',
                             precision='10', name='matrixExporter')
        # Fix the base of the trunk by adding constraints in a region of interest (ROI)
        self.robot.addObject('BoxROI', name='boxROI', box=[[-20, -20, 0], [20, 20, 20]], drawBoxes=False)
        self.robot.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness='1e12')

        self.robot.min_force = [0.] * 8

        actuator_names = ''

        length1 = 10.
        length2 = 2.
        lengthTrunk = 195.

        pullPoint = [[0., length1, 0.], [-length1, 0., 0.], [0., -length1, 0.], [length1, 0., 0.]]
        direction = Vec3(0., length2 - length1, lengthTrunk)
        direction.normalize()
        nbCables = 4
        actuators = self.robot.addChild('actuators')
        for i in range(0, nbCables):
            childname = 'cableL' + str(i)

            theta = 1.57 * i
            q = Quat(0., 0., sin(theta / 2.), cos(theta / 2.))
            position = [[0., 0., 0.]] * 20
            for k in range(0, 20, 2):
                v = Vec3(direction[0], direction[1] * 17.5 * (k / 2) + length1, direction[2] * 17.5 * (k / 2) + 21)
                position[k] = v.rotateFromQuat(q)
                v = Vec3(direction[0], direction[1] * 17.5 * (k / 2) + length1, direction[2] * 17.5 * (k / 2) + 27)
                position[k + 1] = v.rotateFromQuat(q)
            cableL = actuators.addChild(childname)
            cableL.addObject('MechanicalObject', name='meca',
                             position=pullPoint[i] + [pos.toList() for pos in position])
            cableL.addObject('CableConstraint', template='Vec3d', name="cable",
                             hasPullPoint="0",
                             indices=list(range(21)),
                             maxPositiveDisp='70',
                             maxDispVariation="1",
                             valueType='force',
                             minForce=self.robot.min_force[i] * self.robot.dt.value)

            cableL.addObject('BarycentricMapping', name='mapping', mapForces='false', mapMasses='false')
            actuator_names += childname + '/cable,'
            self.actuator_list.append(cableL.cable)

        if all_cables:
            for i in range(0, nbCables):
                childname = 'cableS' + str(i)
                theta = 1.57 * i
                q = Quat(0., 0., sin(theta / 2.), cos(theta / 2.))

                position = [[0., 0., 0.]] * 10
                for k in range(0, 9, 2):
                    v = Vec3(direction[0], direction[1] * 17.5 * (k / 2) + length1, direction[2] * 17.5 * (k / 2) + 21)
                    position[k] = v.rotateFromQuat(q)
                    v = Vec3(direction[0], direction[1] * 17.5 * (k / 2) + length1, direction[2] * 17.5 * (k / 2) + 27)
                    position[k + 1] = v.rotateFromQuat(q)

                cableS = actuators.addChild(childname)
                cableS.addObject('MechanicalObject', name='meca',
                                 position=pullPoint[i] + [pos.toList() for pos in position])
                cableS.addObject('CableConstraint', template='Vec3d', name="cable",
                                 hasPullPoint="0",
                                 indices=list(range(10)),
                                 maxPositiveDisp='40',
                                 maxDispVariation="1",
                                 valueType='force',
                                 minForce=self.robot.min_force[i + 4] * self.robot.dt.value)
                cableS.addObject('BarycentricMapping', name='mapping', mapForces='false', mapMasses='false')
                actuator_names += childname + '/cable,'
                self.actuator_list.append(cableS.cable)

        self.robot.actuator_list = self.actuator_list
        ##########################################
        # Visualization                          #
        ##########################################
        trunkVisu = self.robot.addChild('VisualModel')
        trunkVisu.addObject('MeshSTLLoader', filename=path + "/mesh/trunk.stl")
        trunkVisu.addObject('OglModel', template='Vec3d', color=[1., 1., 1., 0.8])
        trunkVisu.addObject('BarycentricMapping')


class Diamond(TemplateEnvironment):
    def __init__(self, name='Diamond', totalMass=0.5, poissonRatio=0.45, youngModulus=450, rayleighMass=0.1, rayleighStiffness=0.1, dt=0.01,
                 q0=None, scale_mode=1000):
        super(Diamond, self).__init__(name=name, rayleighMass=rayleighMass, rayleighStiffness=rayleighStiffness, dt=dt)

        self.nb_nodes = 1628
        self.gravity = [0., 0., -9810.]
        #self.gravity = [0., 0., 0.]

        # Don't change these values (affects physics)
        rotation = [90., 0.0, 0.0]
        translation = [0.0, 0.0, 35]

        self.robot.min_force = [0, 0, 0, 0]  # Without premultiplication with dt

        self.robot.addObject('MeshVTKLoader', name='loader', filename=path + "/mesh/diamond.vtu", rotation=rotation,
                             translation=translation)

        # Rest position before gravity
        default_rest_position = self.robot.loader.position.toList()

        rest_file = path + "/../examples/diamond/rest.pkl"
        path_rest_file = Path(rest_file)
        if path_rest_file.exists():
            print("Loading Rest File after gravity effect")
            rest_data = load_data(rest_file)
            rest_position_1d = rest_data["rest"]
            rest_position = np.array([rest_position_1d[3*ii:(3*ii+3)] for ii in range(len(rest_position_1d) // 3)])
        else:
            rest_position = copy.deepcopy(default_rest_position)

        initial_position = copy.deepcopy(default_rest_position)

        # Set initial configuration of robot (default is 1000)
        if q0 is not None:
            initial_position = rest_position + scale_mode * q0

        self.robot.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.robot.addObject('TetrahedronSetTopologyModifier')
        self.robot.addObject('TetrahedronSetGeometryAlgorithms')
        self.robot.addObject('MechanicalObject', template='Vec3d', name='tetras', showIndices='false',
                             showIndicesScale='4e-5', position=initial_position, rest_position=default_rest_position)
        self.robot.addObject('UniformMass', totalMass=totalMass, name='mass')
        self.robot.addObject('TetrahedronFEMForceField', template='Vec3d',
                             method='large', name='forcefield',
                             poissonRatio=poissonRatio, youngModulus=youngModulus)
        # self.robot.addObject('GlobalSystemMatrixExporter', exportEveryNumberOfSteps='10', enable='False',
        #                      precision='10', name='matrixExporter')
        # Fix the base of the trunk by adding constraints in a region of interest (ROI)
        self.robot.addObject('BoxROI', name='boxROI', box=[-15, -15, -40, 15, 15, 10], drawBoxes=True)
        self.robot.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness='1e12', name='constraints')

        ##########################################
        # Cable                                 #
        ##########################################
        self.actuatorsParam = [
            {'withName': 'A',
             'withCableGeometry': [[0, 97, 45]],
             'withAPullPointLocation': [0, 10, 30]
             },
            {'withName': 'B',
             'withCableGeometry': [[-97, 0, 45]],
             'withAPullPointLocation': [-10, 0, 30]
             },
            {'withName': 'C',
             'withCableGeometry': [[0, -97, 45]],
             'withAPullPointLocation': [0, -10, 30]
             },
            {'withName': 'D',
             'withCableGeometry': [[97, 0, 45]],
             'withAPullPointLocation': [10, 0, 30]
             }
        ]

        actuators = self.robot.addChild('actuators')

        for i in range(len(self.actuatorsParam)):
            cable = actuators.addChild(self.actuatorsParam[i]['withName'])
            cable.addObject('MechanicalObject', position=self.actuatorsParam[i]['withCableGeometry'])
            cable.addObject('CableConstraint',
                            name='cable',
                            indices=list(range(len(self.actuatorsParam[i]['withCableGeometry']))),
                            pullPoint=self.actuatorsParam[i]['withAPullPointLocation'],
                            valueType='force',
                            hasPullPoint=True,
                            minForce=self.robot.min_force[i] * self.robot.dt.value
                            )

            cable.addObject('BarycentricMapping', name="Mapping", mapForces=False, mapMasses=False)
            self.actuator_list.append(cable.cable)

        self.robot.actuator_list = self.actuator_list

        ##########################################
        # Visualization                          #
        ##########################################
        diamondVisu = self.robot.addChild('VisualModel')
        diamondVisu.addObject('MeshSTLLoader', filename=path + "/mesh/diamond.stl")
        diamondVisu.addObject('OglModel', template='Vec3d', color=[0.7, 0.7, 0.7, 0.7], updateNormals=False)
        diamondVisu.addObject('BarycentricMapping')

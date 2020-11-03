import os
from math import cos
from math import sin

import Sofa.Core
from splib.numerics import Quat, Vec3

from sofacontrol import measurement_models

path = os.path.dirname(os.path.abspath(__file__))


class TemplateEnvironment:
    def __init__(self, name='Template'):
        self.name = name
        self.robot = Sofa.Core.Node(name)
        # set-up solvers
        self.robot.addObject('EulerImplicitSolver', name='odesolver', firstOrder="0", rayleighMass="0.1",
                             rayleighStiffness="0.1")
        self.robot.addObject('SparseLDLSolver', name='preconditioner')
        self.robot.addObject('GenericConstraintCorrection', solverName="preconditioner")
        self.actuator_list = []
        self.nb_nodes = None
        self.gravity = [0., -9810., 0.]  # default
        self.dt = 0.01  # default

    def get_measurement_model(self, nodes=None, pos=True, vel=True):
        if nodes is None:
            return measurement_models.linearModel(range(self.nb_nodes), self.nb_nodes, pos=pos, vel=vel)
        else:
            return measurement_models.linearModel(nodes, self.nb_nodes, pos=pos, vel=vel)


class Trunk(TemplateEnvironment):
    def __init__(self, name='Trunk', all_cables=True):
        super(Trunk, self).__init__(name=name)

        self.nb_nodes = 709

        self.gravity = [0., 0., 9810.]

        self.robot.min_force = [0.] * 8  # Without premultiplication with dt

        self.robot.addObject('MeshVTKLoader', name='loader', filename=path + '/mesh/trunk.vtk')
        self.robot.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.robot.addObject('TetrahedronSetTopologyModifier')
        self.robot.addObject('TetrahedronSetTopologyAlgorithms')
        self.robot.addObject('TetrahedronSetGeometryAlgorithms')
        # Option 1:
        self.robot.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false',
                             showIndicesScale='4e-5')
        # Option 2: Equivalent to option 1 (we believe)
        # self.robot.addObject('MechanicalObject', src='@loader')

        # Gives a mass to the model
        self.robot.addObject('UniformMass', totalMass=0.042)
        # Add a TetrahedronFEMForceField componant which implement an elastic material model solved using the Finite
        # Element Method on tetrahedrons.
        self.robot.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large',
                             poissonRatio=0.45,
                             youngModulus=450)
        # Fix the base of the trunk by adding constraints in a region of interest (ROI)
        self.robot.addObject('BoxROI', name='boxROI', box=[[-20, -20, 0], [20, 20, 20]], drawBoxes=False)
        self.robot.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness='1e12')

        ##########################################
        # Cable                                  #
        ##########################################
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


class Trunk4Cables(Trunk):
    def __init__(self, name='Trunk4Cables'):
        super(Trunk4Cables, self).__init__(name=name, all_cables=False)
        self.robot.min_force = [0, 0, 0, 0]  # Without premultiplication with dt


class Finger(TemplateEnvironment):
    def __init__(self, name='Finger'):
        super(Finger, self).__init__(name=name)

        self.nb_nodes = 158

        self.robot.min_force = [0.]  # Without premultiplication with dt

        self.robot.addObject('MeshVTKLoader', name='loader', filename=path + '/mesh/finger.vtk')
        self.robot.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.robot.addObject('TetrahedronSetTopologyModifier')
        self.robot.addObject('TetrahedronSetTopologyAlgorithms')
        self.robot.addObject('TetrahedronSetGeometryAlgorithms')
        self.robot.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false',
                             showIndicesScale='4e-5')

        self.robot.addObject('UniformMass', totalMass=0.075)

        # Add a TetrahedronFEMForceField componant which implement an elastic material model solved using the Finite Element Method on tetrahedrons.
        self.robot.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large',
                             poissonRatio=0.45,
                             youngModulus=600)

        # Fix the base of the trunk by adding constraints in a region of interest (ROI)
        self.robot.addObject('BoxROI', name='boxROI', box=[[-15, 0, 0], [5, 10, 15]], drawBoxes=False)
        self.robot.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness='1e12')

        ##########################################
        # Cable                                 #
        ##########################################

        #  This creates a new node in the scene. This node is appended to the finger's node.
        actuators = self.robot.addChild('actuators')

        cable = actuators.addChild('cable')

        #  This create a MechanicalObject, a componant holding the degree of freedom of our
        # mechanical modelling. In the case of a cable it is a set of positions specifying
        #  the points where the cable is passing by.
        cable.addObject('MechanicalObject', name='meca',
                        position=(
                                "-17.5 12.5 2.5 " +
                                "-32.5 12.5 2.5 " +
                                "-47.5 12.5 2.5 " +
                                "-62.5 12.5 2.5 " +
                                "-77.5 12.5 2.5 " +

                                "-83.5 12.5 4.5 " +
                                "-85.5 12.5 6.5 " +
                                "-85.5 12.5 8.5 " +
                                "-83.5 12.5 10.5 " +

                                "-77.5 12.5 12.5 " +
                                "-62.5 12.5 12.5 " +
                                "-47.5 12.5 12.5 " +
                                "-32.5 12.5 12.5 " +
                                "-17.5 12.5 12.5 "))

        # Create a CableConstraint object with a name.
        # the indices are referring to the MechanicalObject's positions.
        # The last indice is where the pullPoint is connected.
        cable.addObject('CableConstraint', name="cable",
                        indices=list(range(14)),
                        pullPoint="0.0 12.5 2.5", valueType='force',
                        minForce=self.robot.min_force[0] * self.robot.dt.value)
        # This create a BarycentricMapping. A BarycentricMapping is a key element as it will create a bi-directional link
        #  between the cable's DoFs and the finger's ones so that movements of the cable's DoFs will be mapped
        #  to the finger and vice-versa;
        cable.addObject('BarycentricMapping', name='mapping', mapForces='false', mapMasses='false')

        self.actuator_list.append(cable.cable)

        self.robot.actuator_list = self.actuator_list
        ##########################################
        # Visualization                          #
        ##########################################
        # In Sofa, visualization is handled by adding a rendering model.
        #  Create an empty child node to store this rendering model.
        fingerVisu = self.robot.addChild('VisualModel')

        # Add to this empty node a rendering model made of triangles and loaded from an stl file.
        fingerVisu.addObject('MeshSTLLoader', filename=path + "/mesh/finger.stl")
        fingerVisu.addObject('OglModel', template='Vec3d', color=[1., 1., 1., 0.8])

        # Add a BarycentricMapping to deform rendering model in way that follow the ones of the parent mechanical model.
        fingerVisu.addObject('BarycentricMapping')


class Diamond(TemplateEnvironment):
    def __init__(self, name='Diamond'):
        super(Diamond, self).__init__(name=name)

        self.nb_nodes = 1628
        self.gravity = [0., 0., -9810.]

        rotation = [90, 0.0, 0.0]
        translation = [0.0, 0.0, 35]

        self.robot.min_force = [0, 0, 0, 0]  # Without premultiplication with dt

        self.robot.addObject('MeshVTKLoader', name='loader', filename=path + "/mesh/diamond.vtu", rotation=rotation,
                             translation=translation)
        self.robot.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.robot.addObject('TetrahedronSetTopologyModifier')
        self.robot.addObject('TetrahedronSetTopologyAlgorithms')
        self.robot.addObject('TetrahedronSetGeometryAlgorithms')
        self.robot.addObject('MechanicalObject', template='Vec3d', name='tetras', showIndices='false',
                             showIndicesScale='4e-5')
        self.robot.addObject('UniformMass', totalMass=0.5, name='mass')
        self.robot.addObject('TetrahedronFEMForceField', template='Vec3d',
                             method='large', name='forcefield',
                             poissonRatio=0.45, youngModulus=450)

        # Fix the base of the trunk by adding constraints in a region of interest (ROI)
        self.robot.addObject('BoxROI', name='boxROI', box=[-15, -15, -40, 15, 15, 10], drawBoxes=True)
        self.robot.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness='1e12')

        ##########################################
        # Cable                                 #
        ##########################################
        actuatorsParam = [
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

        for i in range(len(actuatorsParam)):
            cable = actuators.addChild(actuatorsParam[i]['withName'])
            cable.addObject('MechanicalObject', position=actuatorsParam[i]['withCableGeometry'])
            cable.addObject('CableConstraint',
                            name='cable',
                            indices=list(range(len(actuatorsParam[i]['withCableGeometry']))),
                            pullPoint=actuatorsParam[i]['withAPullPointLocation'],
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

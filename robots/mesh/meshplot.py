import numpy as np
import vtk
from scipy.io import savemat
from vtk.util.numpy_support import vtk_to_numpy


def parse_triangles(triangles):
    """
    format: [n0, id0(1), ..., id0(n0),
             n1, id1(1), ..., id1(n1), ...]
    """
    new = []
    i = 0
    while i < len(triangles):
        n = triangles[i]
        triangle = np.zeros(n)
        for j in range(1, n + 1):
            triangle[j - 1] = triangles[i + j]
        new.append(triangle)
        i += n + 1

    triangles = np.empty(len(new), dtype=object)
    for i in range(len(new)):
        triangles[i] = new[i]
    return triangles


def loadvtk(filename):
    """ Loads either vtk or vtu files and outputs the parsed
    data:
    points: (N,3) array of positions of each N node
    triangles: collection of lists of node numbers that make
    up each tetrahedron/triangle
    """
    if filename.split('.')[-1] == 'vtk':
        reader = vtk.vtkDataSetReader()
    elif filename.split('.')[-1] == 'vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        print('Unknown file type.')
        return
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    points = vtk_to_numpy(data.GetPoints().GetData())
    triangles = vtk_to_numpy(data.GetCells().GetData())
    triangles = parse_triangles(triangles)
    return points, triangles


def save2mat(filename, points, triangles):
    """ Save the data to mat file for plotting in matlab"""
    data = {}
    data['points'] = points
    data['triangles'] = triangles
    savemat(filename, data)


if __name__ == '__main__':
    points, triangles = loadvtk('diamond.vtu')
    save2mat('diamond.mat', points, triangles)
    points, triangles = loadvtk('trunk.vtk')
    save2mat('trunk.mat', points, triangles)

import vtk
import numpy as np
from vtk.numpy_interface.dataset_adapter import UnstructuredGrid

def write_vtu_file(points, polyhedrons, pressure, filename='out.vtu'):
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((points.shape[0],1))))
    
    unstructured_grid = UnstructuredGrid(vtk.vtkUnstructuredGrid())
    unstructured_grid.SetPoints(points)
    for polyhedron in polyhedrons:
        face_stream = vtk.vtkIdList()
        face_stream.InsertNextId(len(polyhedron))
        for face in polyhedron:
            face_stream.InsertNextId(len(face))
            for point_id in face:
                face_stream.InsertNextId(point_id)
        unstructured_grid.InsertNextCell(vtk.VTK_POLYHEDRON, face_stream)

    property_name = "Pressure"
    property_data_array = vtk.vtkFloatArray()
    property_data_array.SetName(property_name)
    for value in pressure.flatten():
        property_data_array.InsertNextValue(value)
    point_data = unstructured_grid.GetPointData()
    point_data.SetActiveScalars(property_name)
    point_data.SetScalars(property_data_array)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetDataModeToAscii()
    writer.SetCompressorTypeToNone()
    writer.SetFileName(filename)
    writer.SetInputData(unstructured_grid.VTKObject)
    writer.Write()


def save_vtu_file(P, mesh, *args, **kwargs):
    P = P.real
    write_vtu_file(mesh.points, [mesh.connectivity_list], P)


def get_callbacks():
    return {
        'on_after_solve_2d_helmholtz': [save_vtu_file,],
    }

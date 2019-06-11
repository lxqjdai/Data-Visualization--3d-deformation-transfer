import vtk

filename = "output.obj"
reader = vtk.vtkOBJReader()
reader.SetFileName(filename)

skinMapper = vtk.vtkPolyDataMapper()
skinMapper.SetInputConnection(reader.GetOutputPort())
skinMapper.ScalarVisibilityOff()
skin = vtk.vtkActor()
skin.SetMapper(skinMapper)

# Create a rendering window and renderer
ren1 = vtk.vtkRenderer()
ren1.AddActor(skin)
ren1.SetViewport(0,0,0.5,1)
ren1.SetBackground(255,255,255)
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren1)

filename2 = "result.obj"
reader2 = vtk.vtkOBJReader()
reader2.SetFileName(filename2)

skinMapper2 = vtk.vtkPolyDataMapper()
skinMapper2.SetInputConnection(reader2.GetOutputPort())
skinMapper2.ScalarVisibilityOff()
skin2 = vtk.vtkActor()
skin2.SetMapper(skinMapper2)

# Create a rendering window and renderer
ren2 = vtk.vtkRenderer()
ren2.AddActor(skin2)
ren2.SetViewport(0.5,0,1,1)
renWin.AddRenderer(ren2)

# Create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()
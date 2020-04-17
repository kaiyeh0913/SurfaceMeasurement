# SurfaceMeasurement
Reconstruction backbone for Surface Measurement App with Python for Computational Photography Lab

Create by Kai Yeh

#### Dependencies

Python 3.7x (fully tested) / 2.7x (untested but should work)

Numpy, Scipy, OpenCV

#### Usage

In the reconstruction framework, we include two different methods for different material of the objects. Deflectometry is a surface scanning method that suits for shiny surface and Gradient Illumination suits for diffuse(matt) surface.

python main.py -m [mode: 0-Both, 1-Deflectometry 2-Gradient Illumination] -d [ImgPathDirectory]

output: obj, mtl(mesh), jpg(texture), png(normal map)

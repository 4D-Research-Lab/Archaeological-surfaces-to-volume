# BachelorScriptie
Code for my bachelor thesis project on automated generation of a volumetric mesh from multiple polygonal meshes.

This is a Python 3.7 script made for Blender. Open and run this script in Blender and it will automatically try
to approximate the 3D volume between two or multiple triangular meshes.

## Required modules
The following modules are required to be installed, to run this script:
  - NumPy (To install on Python 3.7: ```python3.7 -m pip install numpy```)
  - plyfile (To install on Python 3.7: ```python3.7 -m pip install plyfile```)
 
Other non-standard modules that are used are:
  - bpy
  - bmesh
  - mathutils

These modules are automatically loaded in Blender's Python environment. This means that they do not need to be installed separately by the user, if the script is loaded and run in Blender.

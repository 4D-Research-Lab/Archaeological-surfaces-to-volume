# 3D volume generation
The code in `vol_generation_script.py` is a Blender script written in Python 3 used to produce a 3D volumetric mesh/object between 2 input meshes. This can be done on more than 2 meshes, for each consecutive pair of meshes (top to bottom) a 3D volume with a certain color is generated. Also the volume of each of this layers and the total volume of all layers is calculated. The purpose of this is to give archeologists a tool to visualize and analyse their data better. The input data is expected to be different stadia of excavation surfaces and have to be already loaded and selected in Blender.

### Example input
![Example input](/Images/before_meshes.png "Exaple input")

### Example output
![Example output](/Images/after_with_original.png "Example output")

### Example of the generated 3D volumetric mesh
![Generated 3D volumetric mesh](/Images/after_no_originals.png "Generated 3D volumetric mesh")

This is made for my bachelor thesis project at University of Amsterdam on automated generation of a volumetric mesh from multiple polygonal meshes.


## Requirements
If you want to execute this script Blender must be installed ([Free to download here](https://www.blender.org/)). In Blender you can load the script and run it, no external packages or modules are needed. The output can also be visualized in Blender.

## Usage
1. Load all your objects into Blender
2. Load this script into Blender (scripting tab)
3. Specifiy which algorithm you want to run and set some variables to your preference (See [User input](#user-input))
4. Select all objects between which you want to generate a volumetric mesh
5. Run the script
6. Visualize and analyze the output

## User input
In the main function (at the bottom of the script) the user can set some variables to create the output they want. Those variables are:
- threshold: the minimum distance in centimeters between the meshes for it to be considered a significant distance, e.g. points with a distance shorter than threshold to the other mesh are considered not to be in between the meshes
- method: specify the type of algorithm you want to run, space- or object-oriented (See [General Information](#general-explanation-of-the-used-algorithms))

For space-oriented:
- primitive: type of volume primitve used, cuboids (cuboid) or tetrahedra (tetra)
- length: length of the volume primitive to be used in centimeters
- width: width of the volume primitive to be used in centimeters
- height: height of the volume primitive to be used in centimeters
- draw: draw the generated 3D volumetric mesh or not
- export_ply_file: export the generated 3D volumetric mesh as PLY-file or not
- ply_file_name: specify the name of the exported PLY-file

## General explanation of the used algorithms
### Space-oriented algorithm
The space-oriented algorithm divides the 3D bounding box in volume primitives of given shape and size. Then for every primitive it is decided if it is part of the volume between the meshes or not. This is determined using the `bpy.types.Object.ray_cast` function. 1 ray is shot straight up ((0, 0, 1)-direction) and 1 ray is shot straight down ((0, 0 -1)-direction). The primitive is said to be between the meshes if:
- the ray shot upwards hits the upper mesh an uneven number of times
- the ray shot downwards hits the lower mesh an uneven number of times
- the distance between the two first hitpoints of the rays is greater than or equal to the threshold value

If a primitive is between the meshes, its volume is added to the total volume and it center is written to a temporary storage file (Downloads/vol_gen_temp_storage.txt). It is stored in a file instead of a list to save RAM memory. After all primitives are looped through, the generated mesh is drawn and the PLY-file is created and exported if those variables are set.

### Object-oriented algorithm
The object-oriented algorithm tries to find the boundary of the volume in between the meshes. It first removes all vertices in the meshes which are too close (distance < threshold) to the other mesh. Then it stitches the 2 meshes together based on the stitching step of the JASNOM algorithm. This step does not work correctly as there are still holes left in the mesh. This holes are then filled with the function `bmesh.ops.contextual_create()`, but unfortunately the end mesh is not a manifold one. This means that the volume calculation is not guaranteed to be correct. This algorithm is thus not working properly and needs to be improved.

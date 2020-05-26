# 3D volume generation
The code in `vol_generation_script.py` is a Blender script written in Python 3 used to produce a 3D volumetric mesh/object between 2 input meshes. This can be done on more than 2 meshes, for each consecutive pair of meshes (top to bottom) a 3D volume with a certain color is generated. Also the volume of each of this layers and the total volume of all layers is calculated. The purpose of this is to give archeologists a tool to visualize and analyse their data better. The input data is expected to be different stadia of excavation surfaces and have to be already loaded and selected in Blender.

### Example input
<img src="/Images/before_meshes.png" alt="Example input" width="600">

### Example output
<img src="/Images/before_meshes.png" alt="Example output" width="600">

### Example of the generated 3D volumetric mesh
<img src="/Images/before_meshes.png" alt="Generated 3D volumetric mesh" width="600">

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

#### Important notes!
**Removal of generated objects**: The script uses specific names for the generated volumes. If the script is executed it checks if any objects with this name are present and removes them if they are. The names used are:
- triangle_mesh_volume: for objects created with the object-oriented algorithm
- cubic_volume: for objects created with the cuboid space-oriented algorithm
- tetrahedron_volume: for objects created with the tetrahedra space-oriented algorithm
Thus if you happen to have loaded objects in Blender with names that starts with any of the above names, they will be removed by the script. Also if you change the names of the generated objects they will not be removed by the script.

**Exporting the PLY-file**: PLY-files are exported if the `export_ply_file` variable is set to True to the directory specified with the `path_to_file` variable. This is the path from the users home directory. This path has to be valid with directories that already exist on the user's file system. If not the script will fail and give an error.

**Temporary storage file**: this file (`vol_gen_temp_storage.txt`) is always created in the home directory of the user. It will be removed if the script executes well, but will not when the script fails. If a file with this name already exists in the home directory it will be overwritten by the script and thus it would be best to remove or rename that file.

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
- path_to_file: specify the path to the directory where the PLY-file must be stored
- ply_file_name: specify the name of the exported PLY-file

## General explanation of the used algorithms
### Space-oriented algorithm
The space-oriented algorithm divides the 3D bounding box in volume primitives of given shape and size. Then for every primitive it is decided if it is part of the volume between the meshes or not. This is determined using the `bpy.types.Object.ray_cast` function. 1 ray is shot straight up ((0, 0, 1)-direction) and 1 ray is shot straight down ((0, 0 -1)-direction). The primitive is said to be between the meshes if:
- the ray shot upwards hits the upper mesh an uneven number of times
- the ray shot downwards hits the lower mesh an uneven number of times
- the distance between the two first hitpoints of the rays is greater than or equal to the threshold value

If a primitive is between the meshes, its volume is added to the total volume and it center is written to a temporary storage file (Downloads/vol_gen_temp_storage.txt). It is stored in a file instead of a list to save RAM memory. After all primitives are looped through, the generated mesh is drawn and the PLY-file is created and exported if those variables are set.

### Object-oriented algorithm
The object-oriented algorithm tries to find the boundary of the volume in between the meshes. It first removes all vertices in the meshes which are too close (distance < threshold) to the other mesh. Then it stitches the 2 meshes together based on the stitching step of the JASNOM algorithm ([JASNOM paper](https://kilthub.cmu.edu/articles/Joint_Alignment_and_Stitching_of_Non_Overlapping_Meshes/6606662)). This step does not work correctly as there are still holes left in the mesh. This holes are then filled with the function `bmesh.ops.contextual_create()`, but unfortunately the end mesh is not a manifold one. This means that the volume calculation is not guaranteed to be correct. This algorithm is thus not working properly and needs to be improved.

## Customization
You can customize the space-oriented algorithm to your own liking if you want to. For example you can change the way it is decided if the volume primitve is between the meshes or add a whole new volume primitive used to divide the space and make up the generated volume.

#### Change conditions to be between meshes
If you want to change this, you just have to alter the function `decide_in_volume()`, which now has the function parameters:
- upper_mesh: index of the upper mesh in D.objects list (D = bpy.data)
- lower_mesh: index of the lower mesh in D.objects list
- center: tuple with x-, y- and z-coordinate of the center of the primitive
- max_distance: maximum distance they ray can travel to find a hitpoint (set to be the height of the bounding box)
- threshold: distance in centimeters between the meshes for it to be considered a significant distance

Some ideas for the condition:
- Distance from center to upper hitpoint >= 1/2 threshold and distance from center to lower hitpoint >= 1/2 threshold
- Instead of center test if all vertices of the primitive are between the meshes and there distance to the hitpoints or a percentage of vertices (e.g. 75%)
- Check for even more points inside the primitiveif they are between the meshes and what the distance between the hitpoints is

#### Add new volume primitive
The function `make_point_list(minx, maxx, miny, maxy, minz, maxz, l, w, h)` loops over the bounding box (minx, maxx, miny, maxy, minz, maxz) and divides it in cuboid of length *l*, width *w* and height *h*. It returns the centers of the cuboids. Thus it is easiest to add a new primitive that subdivides a cuboid in equal volume primitives (for example 6 pyramids by simply using the 4 body diagonals of the cuboid). 

Step to execute for adding a primitive called *x*:
- in `space_oriented_volume_between_meshes`, `set_vol_primitive`, `add_primitive` and `finish_bmesh` add a if condition for your primitive (`elif primitive == "x"`).
- in `set_vol_primitive` return the volume of 1 *x*-primitive in cubic centimeters.
- add a function `process_x` that loops over all *x*-primitives in a cuboid with center (x, y, z) and decides if it is between the meshes. If it is add a line to the temporary storage file with its center, a rgb-value and a layer index. Return the sum of the volume of all primitves in this cuboid that are between the meshes.
- in `space_oriented_volume_between_meshes` call the `process_x` function.
- in `add_primitive` add the vertices and faces of the primitive with center (x, y, z) to the bmesh *cur_bm*. Also given are the **cuboid** size (length x width x height) and the left bottom corner of the bounding box (minx, miny, minz). For this you have to write the functions `make_x_verts` (returns tuples of floats as vertices) and `make_x_verts_faces` (returns tuples of BMVerts as faces).
- in `finish_bmesh` you just have to give a name for the generated volume made out of *x*-primitives, both for the mesh and object (for example 'x_volume').
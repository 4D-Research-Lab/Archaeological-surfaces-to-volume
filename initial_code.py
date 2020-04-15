# This is a script made to calculate the volume between different surfaces
# of archaeological excavation grounds. It implements different methods to do 
# so. The surfaces are triangle meshes.
#
# Made by Bram Dekker (11428279) of the Univeristy of Amsterdam as a bachelor
# thesis. 

import bpy
import bmesh
import time
import math
import numpy as np

time_start = time.time()

C = bpy.context
D = bpy.data
FILE_NAMES = ["Halos_trench_10072019.obj", "Halos_trench_15072019.obj",
              "Halos_trench_19072019.obj"]
PATH_TO_DIR = "/home/bram/Documents/Scriptie/dataset/"


def init():
    """Initialize the scene by removing the cube, camera and light and import 
    and orientate the trench objects."""
    clear_scene()
    import_objects()
    rotate_objects()


def clear_scene():
    """Removing the initial cube, camera and light"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for m in D.meshes:
        D.meshes.remove(m)

    
def import_objects():
    """Import the objects for which the volume has to be calculated."""
    for name in FILE_NAMES:
        file_path = "".join([PATH_TO_DIR, name])
        bpy.ops.import_scene.obj(filepath=file_path)


def rotate_objects():
    """Rotate the objects back to their normal form, no rotation at all"""
    for name in D.objects:
        name.rotation_euler = (0.0, 0.0, 0.0)


def find_minmax():
    """Find the minimum and maximum x, y and z-coordinates."""
    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf
    for obj in bpy.data.meshes:
        for v in obj.vertices:
            if v.co[0] < min_x:
                min_x = v.co[0]
            if v.co[0] > max_x:
                max_x = v.co[0]
            if v.co[1] < min_y:
                min_y = v.co[1]
            if v.co[1] > max_y:
                max_y = v.co[1]            
            if v.co[2] < min_z:
                min_z = v.co[2]
            if v.co[2] > max_z:
                max_z = v.co[2]

    print("Min x, max x: {min_x}, {max_x}\nMin y, max y: {min_y}, {max_y}\n"
          "Min z, max z: {min_z}, {max_z}".format(min_x=min_x, max_x=max_x,
          min_y=min_y, max_y=max_y, min_z=min_z, max_z=max_z))

    return min_x, max_x, min_y, max_y, min_z, max_z


def translate_to_origin(min_x, max_x, min_y, max_y, min_z, max_z):
    """Translate the objects to around the origin to get better insight."""
    x_trans = -(min_x + (max_x - min_x) / 2)
    y_trans = -(min_y + (max_y - min_y) / 2)
    z_trans = -(min_z + (max_z - min_z) / 2)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.transform.translate(value=(x_trans, y_trans, z_trans))
    bpy.ops.object.transform_apply()
    bpy.ops.object.select_all(action="DESELECT")

    return x_trans, y_trans, z_trans


def main():
    """Execute the script."""
    init()
    minx, maxx, miny, maxy, minz, maxz = find_minmax()

    difx, dify, difz = translate_to_origin(minx, maxx, miny, maxy, minz, maxz)
    
    # Update the new minimum and maximum coordinates.
    minx, maxx = minx + difx, maxx + difx
    miny, maxy = miny + dify, maxy + dify
    minz, maxz = minz + difz, maxz + difz

    # TODO: Different methods to make 3D models of the ground volume.
    # D.objects[0].select_set(True) # Selects object [0]
    # bpy.context.selected_objects[0].ray_cast
    # bpy.context.selected_objects[0].closest_point_on_mesh
    # D.objects[0].ray_cast or D.objects[0].closest_point_on_mesh

    print("Min x, max x: {min_x}, {max_x}\nMin y, max y: {min_y}, {max_y}\n"
          "Min z, max z: {min_z}, {max_z}".format(min_x=minx, max_x=maxx,
          min_y=miny, max_y=maxy, min_z=minz, max_z=maxz))
    
    print("Script Finished: %.4f sec" % (time.time() - time_start))


if __name__ == "__main__":
    main()    

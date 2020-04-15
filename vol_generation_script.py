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
# FILE_NAMES = ["Halos_trench_20191007.obj", "Halos_trench_20191507.obj"]
FILE_NAMES = ["Halos_trench_20191007.obj", "Halos_trench_20191507.obj",
              "Halos_trench_20191907.obj",
              "Halos_trench_Original_Surface(rough).obj"]
            #   "Halos_trench_20181907_63.obj"]
PATH_TO_DIR = "/home/bram/Documents/Scriptie/dataset_2/"
RED = (1, 0, 0, 1)
GREEN = (0, 1, 0, 1)
BLUE = (0, 0, 1, 1)
GREY = (0.5, 0.5, 0.5, 1)
BLACK = (0, 0, 0, 1)
WHITE = (1, 1, 1, 1)


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


def ordered_meshes():
    """Return the list of indices in de bpy.data.objects list of all
    (highest avg z-coordinate of vertices) objects from high to low."""
    mesh_list = []

    for i, obj in enumerate(D.objects):
        cur_avg = np.average([vt.co[2] for vt in obj.data.vertices.values()])
        mesh_list.append((cur_avg, i))

    return [elem[1] for elem in sorted(mesh_list, reverse=True)]


def dist_btwn_pts(p1, p2):
    """Calculate the distance between 2 points in 3D space"""
    x_dif, y_dif, z_dif = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    return math.sqrt(x_dif**2 + y_dif**2 + z_dif**2)


def center_tetrahedron(v1, v2, v3, v4):
    """Return the center of a tetrahedron with vertices v1, v2, v3 and v4."""
    s = tuple(map(sum, zip(v1, v2, v3, v4)))
    return tuple(el / 4 for el in s)


def voxelize(size, scale, voxel_locations, rgba):
    """Add voxels to the scene to visualize the volume."""
    color = bpy.data.materials.new("Layer_color")

    for loc in voxel_locations:
        bpy.ops.mesh.primitive_cube_add(size=size, location=loc)
        bpy.ops.transform.resize(value=scale)
        C.active_object.data.materials.append(color)
        C.object.active_material.diffuse_color = rgba


def reduce_boundingbox(mesh1, mesh2, threshold, minx, maxx, miny, maxy, maxz):
    """Reduce the bounding box for 2 meshes so that only (x,y)-positions with
    relatively large vertical distance are in it. Mesh1 lays above mesh2."""
    stepsize = 0.01
    c_height = maxz + 1
    cur_minx, cur_maxx = minx, maxx
    minx_found, maxx_found = False, False
    cur_miny, cur_maxy = miny, maxy
    miny_found, maxy_found = False, False

    while not (minx_found and maxx_found):
        for y in np.linspace(miny, maxy, 16):
            if not minx_found:
                min1_ray = D.objects[mesh1].ray_cast((cur_minx, y, c_height),
                                                     (0, 0, -1))
                min2_ray = D.objects[mesh2].ray_cast((cur_minx, y, c_height),
                                                     (0, 0, -1))
                if dist_btwn_pts(min1_ray[1], min2_ray[1]) >= threshold:
                    minx_found = True
                    minx = cur_minx

            if not maxx_found:
                max1_ray = D.objects[mesh1].ray_cast((cur_maxx, y, c_height),
                                                     (0, 0, -1))
                max2_ray = D.objects[mesh2].ray_cast((cur_maxx, y, c_height),
                                                     (0, 0, -1))
                if dist_btwn_pts(max1_ray[1], max2_ray[1]) >= threshold:
                    maxx_found = True
                    maxx = cur_maxx

        cur_minx += stepsize
        cur_maxx -= stepsize

    while not (miny_found and maxy_found):
        for x in np.linspace(minx, maxx, 16):
            if not miny_found:
                min1_ray = D.objects[mesh1].ray_cast((x, cur_miny, c_height),
                                                     (0, 0, -1))
                min2_ray = D.objects[mesh2].ray_cast((x, cur_miny, c_height),
                                                     (0, 0, -1))
                if dist_btwn_pts(min1_ray[1], min2_ray[1]) >= threshold:
                    miny_found = True
                    miny = cur_miny

            if not maxy_found:
                max1_ray = D.objects[mesh1].ray_cast((x, cur_maxy, c_height),
                                                     (0, 0, -1))
                max2_ray = D.objects[mesh2].ray_cast((x, cur_maxy, c_height),
                                                     (0, 0, -1))
                if dist_btwn_pts(max1_ray[1], max2_ray[1]) >= threshold:
                    maxy_found = True
                    maxy = cur_maxy

        cur_miny += stepsize
        cur_maxy -= stepsize

    # print("New bounding box: minx %.2f, maxx %.2f, miny %.2f, maxy %.2f" %
    #       (minx, maxx, miny, maxy))
    return minx, maxx, miny, maxy


def space_oriented_algorithm(num_x, num_y, num_z, threshold, minx, maxx, miny,
                             maxy, minz, maxz, primitive="voxel"):
    """Space-oriented algorithm to make a 3D model out of 2 trianglar
    meshes."""
    tot_volume = 0
    rbgas = [RED, GREEN, BLUE, BLACK]
    max_distance = maxz - minz
    high_low = ordered_meshes()
    for upper_mesh, lower_mesh in zip(high_low[:-1], high_low[1:]):
        volume_primitives = []
        cur_volume = 0

        # Determine primitive size
        l, w, h = ((maxx - minx) / num_x, (maxy - miny) / num_y,
                   (maxz - minz) / num_z)
        size = min(l, w, h)
        scale = (l / size, w / size, h / size)
        if primitive == "voxel":
            vol_primitive = w * l * h
        elif primitive == "tetra":
            vol_primitive = (w * l * h) / 6

        minx, maxx, miny, maxy = reduce_boundingbox(
            upper_mesh, lower_mesh, threshold, minx, maxx, miny, maxy, maxz)

        # Go over every primitive in bounding box space
        for z in np.arange(minz, maxz, h):
            for y in np.arange(miny, maxy, w):
                for x in np.arange(minx, maxx, l):
                    if primitive == "voxel":
                        center = (x+l/2, y+w/2, z+h/2)

                        # Cast ray up to upper mesh and ray down to lower mesh
                        res_up = D.objects[upper_mesh].ray_cast(
                            center, (0, 0, 1), distance=max_distance)
                        res_down = D.objects[lower_mesh].ray_cast(
                            center, (0, 0, -1), distance=max_distance)

                        # If both rays hit the mesh, centroid is in between the
                        # meshes.
                        # Decide if primitive is part of the volume.
                        tot_dist = (dist_btwn_pts(res_up[1], center) +
                                    dist_btwn_pts(res_down[1], center))
                        if (res_up[0] and res_down[0] and
                                dist_btwn_pts(res_up[1], center) >= (h / 2) and
                                dist_btwn_pts(res_down[1], center) >= (h / 2)):
                            volume_primitives.append(center)
                            cur_volume += vol_primitive
                    elif primitive == "tetra":
                        v1 = (x, y, z)
                        v2 = (x + l, y + w, z + h)
                        other_vs = [(x+l, y, z), (x+l, y+w, z), (x, y+w, z),
                                    (x, y+w, z+h), (x, y, z+h), (x+l, y, z+h),
                                    (x+l, y, z)]
                        for v3, v4 in zip(other_vs[:-1], other_vs[1:]):
                            center = center_tetrahedron(v1, v2, v3, v4)
                            dist_up, dist_down = h / 4, h / 4
                            if v3 == other_vs[1] or v4 == other_vs[1]:
                                dist_up = h / 2
                            elif v3 == other_vs[4] or v4 == other_vs[4]:
                                dist_down = h / 2

                            # Cast ray up to upper mesh and down to lower mesh
                            res_up = D.objects[upper_mesh].ray_cast(
                                center, (0, 0, 1), distance=max_distance)
                            res_down = D.objects[lower_mesh].ray_cast(
                                center, (0, 0, -1), distance=max_distance)

                            # If both rays hit the mesh, centroid is in between
                            # the meshes.
                            # Decide if primitive is part of the volume.
                            tot_dist = (dist_btwn_pts(res_up[1], center) +
                                        dist_btwn_pts(res_down[1], center))
                            if (res_up[0] and res_down[0] and dist_btwn_pts(
                                    res_up[1], center) >= dist_up and
                                    dist_btwn_pts(res_down[1], center) >=
                                    dist_down):
                                volume_primitives.append((v1, v2, v3, v4))
                                cur_volume += vol_primitive
        voxelize(size, scale, volume_primitives, rbgas[upper_mesh])
        print("The difference in volume is %.2f m3." % cur_volume)
        tot_volume += cur_volume

    # for obj in [o for o in D.objects if not o.name.startswith("C")]:
    #     obj.hide_viewport = True

    return tot_volume


def main():
    """Execute the script."""
    init()
    minx, maxx, miny, maxy, minz, maxz = find_minmax()

    difx, dify, difz = translate_to_origin(minx, maxx, miny, maxy, minz, maxz)

    # Update the new minimum and maximum coordinates.
    minx, maxx = minx + difx, maxx + difx
    miny, maxy = miny + dify, maxy + dify
    minz, maxz = minz + difz, maxz + difz

    print("Min x, max x: {min_x}, {max_x}\nMin y, max y: {min_y}, {max_y}\n"
          "Min z, max z: {min_z}, {max_z}".format(
              min_x=minx, max_x=maxx, min_y=miny, max_y=maxy, min_z=minz,
              max_z=maxz))

    threshold = 0.3
    primitive = "voxel"
    num_x, num_y, num_z = 20, 20, 10
    volume = space_oriented_algorithm(num_x, num_y, num_z, threshold, minx,
                                      maxx, miny, maxy, minz, maxz,
                                      primitive=primitive)
    print("The difference in volume is %.2f m3." % volume)

    print("Script Finished: %.4f sec" % (time.time() - time_start))


if __name__ == "__main__":
    main()

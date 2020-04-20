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

C = bpy.context
D = bpy.data
# FILE_NAMES = ["Halos_trench_Original_Surface(rough).obj",
            #   "Halos_trench_20191007.obj"]

FILE_NAMES = ["Halos_trench_20191007.obj", "Halos_trench_20191507.obj"]
# FILE_NAMES = ["Halos_trench_20191007.obj", "Halos_trench_20191507.obj",
#               "Halos_trench_20191907.obj",
#               "Halos_trench_Original_Surface(rough).obj"]
            #   "Halos_trench_20181907_63.obj"]
PATH_TO_DIR = "/home/bram/Documents/Scriptie/dataset_2/"
RED = (1, 0, 0, 1)
GREEN = (0, 1, 0, 1)
BLUE = (0, 0, 1, 1)
GREY = (0.5, 0.5, 0.5, 1)
BLACK = (0, 0, 0, 1)
WHITE = (1, 1, 1, 1)
RGBAS = [RED, GREEN, BLUE, GREY, BLACK, WHITE]
MONKEY = bpy.ops.mesh.primitive_monkey_add
TORUS = bpy.ops.mesh.primitive_torus_add
CYL = bpy.ops.mesh.primitive_cylinder_add
CONE = bpy.ops.mesh.primitive_cone_add
CUBE = bpy.ops.mesh.primitive_cube_add
UVSPHERE = bpy.ops.mesh.primitive_uv_sphere_add
ICOSPHERE = bpy.ops.mesh.primitive_ico_sphere_add
B_DIAMOND = bpy.ops.mesh.primitive_brilliant_add
TEAPOT = bpy.ops.mesh.primitive_teapot_add
TORUSKNOT = bpy.ops.mesh.primitive_torusknot_add
STAR = bpy.ops.mesh.primitive_star_add
TETRA = bpy.ops.mesh.primitive_solid_add
TEST_VOLS = {"mo": 2255372.6832, "to": 1174735.7603, "cy": 6242890.2415,
             "co": 2080963.4138, "cu": 8000000, "uv": 4121941.1484,
             "ic": 3658712.0869, "tk": 20238706.8905, "st": 734731.5605,
             "te": 513200.2638, "tp": 25297666.767, "bd": 1638237.8412}


def init(test):
    """Initialize the scene by removing the cube, camera and light and import
    and orientate the trench objects."""
    clear_scene()
    if not test:
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


def bounding_box():
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


def update_minmax(minx, maxx, miny, maxy, minz, maxz, difx, dify, difz):
    """Update the minimum and maximum values based on the translation to the
    origin."""
    minx, maxx = minx + difx, maxx + difx
    miny, maxy = miny + dify, maxy + dify
    minz, maxz = minz + difz, maxz + difz
    return minx, maxx, miny, maxy, minz, maxz


def print_bbox(minx, maxx, miny, maxy, minz, maxz):
    """Print the minimum and maximum values of the data to gain a better
    understanding of the data."""
    print("Min x, max x: {min_x}, {max_x}\nMin y, max y: {min_y}, {max_y}"
          "\nMin z, max z: {min_z}, {max_z}".format(
              min_x=minx, max_x=maxx, min_y=miny, max_y=maxy, min_z=minz,
              max_z=maxz))


def print_total_volume(volume):
    """Print the total volume."""
    print("The total volume is %.2f cm3 or %.2f m3.\n" %
          (volume, volume / 1000000))


def print_num_primitives(num_x, num_y, num_z):
    """Print the number of primitives in all directions."""
    print("Number of primitives in x-direction: %d, y-direction: %d and "
          "z-direction: %d" % (num_x, num_y, num_z))


def print_primitive_size(length, width, height):
    """Print the size of the primitive used."""
    print("Primitive size: %.2f x %.2f x %.2f" % (length, width, height))

def print_test_results(prim, vol, obj_vol, vol_dif):
    """Print the test results of one particular object and method."""
    print("Calculated volume for space-oriented %s algorithm: %.4f ("
          "exact volume is %.4f)" % (prim, vol, obj_vol))
    print("The difference is thus %.4f or %.2f percent" %
          (vol_dif, 100 * vol_dif / vol))


def ordered_meshes():
    """Return the list of indices in de bpy.data.objects list of all
    objects from high to low (highest avg z-coordinate of vertices)."""
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


def make_voxel_verts(origin, length, width, height):
    """Make a list of vertices for given origin and sizes."""
    v0 = origin
    v1 = (origin[0] + length, origin[1], origin[2])
    v2 = (origin[0], origin[1] + width, origin[2])
    v3 = (origin[0] + length, origin[1] + width, origin[2])
    v4 = (origin[0], origin[1], origin[2] + height)
    v5 = (origin[0] + length, origin[1], origin[2] + height)
    v6 = (origin[0], origin[1] + width, origin[2] + height)
    v7 = (origin[0] + length, origin[1] + width, origin[2] + height)
    return [v0, v1, v2, v3, v4, v5, v6, v7]


def make_voxel_faces():
    """Make a list of faces for given vertices of a cube."""
    f1 = (0, 1, 3, 2)
    f2 = (4, 5, 7, 6)
    f3 = (0, 2, 6, 4)
    f4 = (1, 3, 7, 5)
    f5 = (0, 1, 5, 4)
    f6 = (2, 3, 7, 6)
    return [f1, f2, f3, f4, f5, f6]


def draw_cubes(origins, length, width, height, rgba):
    """Draw cubes from given origins, size and color"""
    # Specify the color and faces for the new meshes.
    color = D.materials.new("Layer_color")
    faces = make_voxel_faces()

    for o in origins:
        # Make the vertices of the new mesh.
        verts = make_voxel_verts(o, length, width, height)

        # Create a new mesh.
        mesh = D.meshes.new("cube")
        mesh.from_pydata(verts, [], faces)
        mesh.validate()
        mesh.update()

        # Create new object and give it data.
        new_object = D.objects.new("cube", mesh)
        new_object.data = mesh

        # Put object in the scene.
        C.collection.objects.link(new_object)
        C.view_layer.objects.active = new_object
        new_object.select_set(True)

        # Give the object the color rgba and deselect it.
        C.active_object.data.materials.append(color)
        C.object.active_material.diffuse_color = rgba
        new_object.select_set(False)


def draw_tetrahedra(vertices, rgba):
    """Draw tetrahedra from given vertices and color."""
    # Specify the color and faces for the new meshes.
    color = D.materials.new("Layer_color")
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

    for verts in vertices:
        # Create new mesh structure.
        mesh = D.meshes.new("tetrahedron")
        mesh.from_pydata(verts, [], faces)
        mesh.validate()
        mesh.update()

        # Create new object and give it data.
        new_object = D.objects.new("tetrahedron", mesh)
        new_object.data = mesh

        # Put object in the scene.
        C.collection.objects.link(new_object)
        C.view_layer.objects.active = new_object
        new_object.select_set(True)

        # Give the object the color rgba and deselect it.
        C.active_object.data.materials.append(color)
        C.object.active_material.diffuse_color = rgba
        new_object.select_set(False)


def cast_down_ray(mesh, var, cur, h, dr="x"):
    """Cast a ray from (x,y,h) or (y,x,h) in direction (dirx, diry, dirz)"""
    if dr == "x":
        result = D.objects[mesh].ray_cast((cur, var, h), (0, 0, -1))
    elif dr == "y":
        result = D.objects[mesh].ray_cast((var, cur, h), (0, 0, -1))

    return result


def find_direction_minmax(msh1, msh2, threshold, minc, maxc, maxz,
                          stepsize, cast_hgt, cur_min, cur_max, dr="x"):
    """Find the minimum and maximum coordinate in a specific direction for
    which the threshold is exceeded."""
    # Initialize minimum and maximum found to false.
    min_found, max_found = False, False

    while not (min_found and max_found):
        # For variable c value, check if current value and this c is a position
        # that exceeds threshold.
        for c in np.linspace(minc, maxc, 50):
            if not min_found:
                # Cast rays to both meshes from cast height downward.
                min1_ray = cast_down_ray(msh1, cur_min, c, cast_hgt, dr=dr)
                min2_ray = cast_down_ray(msh2, cur_min, c, cast_hgt, dr=dr)

                # Check if both rays hit the meshes and check distance between
                # the hit locations.
                if (min1_ray[0] and min2_ray[0] and
                        dist_btwn_pts(min1_ray[1], min2_ray[1]) >= threshold):
                    min_found = True
                    min_c = cur_min

            if not max_found:
                # Cast rays to both meshes from cast height downward.
                max1_ray = cast_down_ray(msh1, cur_max, c, cast_hgt, dr=dr)
                max2_ray = cast_down_ray(msh2, cur_max, c, cast_hgt, dr=dr)

                # Check if both rays hit the meshes and check distance between
                # the hit locations.
                if (max1_ray[0] and max2_ray[0] and
                        dist_btwn_pts(max1_ray[1], max2_ray[1]) >= threshold):
                    max_found = True
                    max_c = cur_max

        # Increase minimum and decrease maximum with every step.
        cur_min += stepsize
        cur_max -= stepsize

    return min_c, max_c


def reduce_boundingbox(mesh1, mesh2, threshold, minx, maxx, miny, maxy, maxz):
    """Reduce the bounding box for 2 meshes so that only (x,y)-positions with
    relatively large vertical distance are in it. Mesh1 lays above mesh2."""
    # Set the stepsize, cast height and current minimum and maximum values.
    stepsize = 0.01
    cast_hgt = maxz + 1
    cur_minx, cur_maxx = minx, maxx
    cur_miny, cur_maxy = miny, maxy

    # Find new minimum and maximum x-values based on the threshold.
    minx, maxx = find_direction_minmax(
        mesh1, mesh2, threshold, miny, maxy, maxz, stepsize, cast_hgt,
        cur_minx, cur_maxx, dr="x")

    # Find new minimum and maximum y-values based on the threshold.
    miny, maxy = find_direction_minmax(
        mesh1, mesh2, threshold, minx, maxx, maxz, stepsize, cast_hgt,
        cur_miny, cur_maxy, dr="y")

    print("New bounding box: minx %.2f, maxx %.2f, miny %.2f, maxy %.2f" %
          (minx, maxx, miny, maxy))
    return minx, maxx, miny, maxy


def set_vol_primitive(length, width, height, primitive="voxel"):
    """Calculate the volume of the specified primitive in cubic centimetres."""
    if primitive == "voxel":
        vol_primitive = length * width * height * 1000000
    elif primitive == "tetra":
        vol_primitive = 1000000 * (length * width * height) / 6

    return vol_primitive


def decide_in_volume(upper_mesh, lower_mesh, center, max_distance):
    """Decide if the primitive on the given position belongs to the volume
    between the meshes."""
    # Cast ray up to upper mesh and ray down to lower mesh.
    res_up = D.objects[upper_mesh].ray_cast(
        center, (0, 0, 1), distance=max_distance)
    res_down = D.objects[lower_mesh].ray_cast(
        center, (0, 0, -1), distance=max_distance)

    # If both rays hit the mesh, centroid is in between the meshes.
    if (res_up[0] and res_down[0]):
        in_volume = True
    else:
        in_volume = False

    return in_volume


def make_point_list(minx, maxx, miny, maxy, minz, maxz, l, w, h):
    """Make a list of cartesian points between boundaries."""
    xs = np.arange(minx, maxx, l)
    ys = np.arange(miny, maxy, w)
    zs = np.arange(minz, maxz, h)
    return [(x, y, z) for x in xs for y in ys for z in zs] 


def space_oriented_volume_between_meshes(
        num_x, num_y, num_z, threshold, minx, maxx, miny, maxy, minz, maxz,
        upper_mesh, lower_mesh, max_distance,  primitive="voxel", test=False):
    """Calculate the volume and make a 3D model of the volume between 2
    meshes."""
    # Initialize volume primitives list and the volume.
    volume_primitives = []
    cur_volume = 0

    # Reduce the bounding box, if it is not a test.
    if not test:
        minx, maxx, miny, maxy = reduce_boundingbox(
            upper_mesh, lower_mesh, threshold, minx, maxx, miny, maxy,
            maxz)

    # Determine primitive size
    l, w, h = ((maxx - minx) / num_x, (maxy - miny) / num_y,
               (maxz - minz) / num_z)

    # Set volume of the primitive used.
    vol_primitive = set_vol_primitive(l, w, h, primitive=primitive)

    # Go over every primitive in bounding box space
    points = make_point_list(minx, maxx, miny, maxy, minz, maxz, l, w, h)

    for x, y, z in points:
        if primitive == "voxel":
            # Determine center of voxel.
            center = (x+l/2, y+w/2, z+h/2)

            # Decide if voxel is part of the volume.
            in_volume = decide_in_volume(upper_mesh, lower_mesh,
                                            center, max_distance)
            if in_volume:
                volume_primitives.append((x, y, z))
                cur_volume += vol_primitive
        elif primitive == "tetra":
            # Get all vertices for the 6 tetrahedra.
            v1, v2 = (x, y, z), (x + l, y + w, z + h)
            other_vs = [(x+l, y, z), (x+l, y+w, z), (x, y+w, z),
                        (x, y+w, z+h), (x, y, z+h), (x+l, y, z+h),
                        (x+l, y, z)]
            for v3, v4 in zip(other_vs[:-1], other_vs[1:]):
                # Determine center of tetrahedron.
                center = center_tetrahedron(v1, v2, v3, v4)

                # Decide if the tetrahedron is part of the volume.
                in_volume = decide_in_volume(upper_mesh, lower_mesh,
                                                center, max_distance)
                if in_volume:
                    volume_primitives.append((v1, v2, v3, v4))
                    cur_volume += vol_primitive

    return cur_volume, l, w, h, volume_primitives


def space_oriented_algorithm(
        meshes, num_x, num_y, num_z, threshold, minx, maxx, miny, maxy, minz,
        maxz, primitive="voxel", test=False):
    """Space-oriented algorithm to make a 3D model out of 2 trianglar
    meshes."""
    # Initialize the total volume and maximum vertical distance.
    tot_volume = 0
    max_distance = maxz - minz

    # For all ordered mesh pairs from top to bottom, run the space-oriented
    # algorithm to get volume and list of primitives.
    for upper_mesh, lower_mesh in zip(meshes[:-1], meshes[1:]):
        # Execute space-oriented algorithm between 2 meshes.
        volume, l, w, h, vol_prims = space_oriented_volume_between_meshes(
            num_x, num_y, num_z, threshold, minx, maxx, miny, maxy, minz, maxz,
            upper_mesh, lower_mesh, max_distance, primitive=primitive,
            test=test)

        # Print the size of the primitives used.
        print_primitive_size(l, w, h)

        # Update total volume.
        tot_volume += volume

        # Draw the 3D model based on the type of primitive.
        if primitive == "voxel":
            draw_cubes(vol_prims, l, w, h, RGBAS[upper_mesh])
        elif primitive == "tetra":
            draw_tetrahedra(vol_prims, RGBAS[upper_mesh])

    return tot_volume


def vol_test(num_x, num_y, num_z, threshold, methods=["tetra"]):
    """Test the self-made algorithm against Blenders built-in for closed
    meshes. Volumes are here calculated in cm^3."""
    # Set the test objects which are mesh primitives in Blender.
    object_list = [
        ("mo", MONKEY), ("to", TORUS), ("cy", CYL), ("co", CONE), ("cu", CUBE),
        ("uv", UVSPHERE), ("ic", ICOSPHERE), ("bd", B_DIAMOND), ("tp", TEAPOT),
        ("tk", TORUSKNOT), ("st", STAR), ("te", TETRA)]

    for prim in methods:
        for name, obj in object_list:
            # Create the object and get its volume.
            obj()
            obj_vol = TEST_VOLS[name]

            minx, maxx, miny, maxy, minz, maxz = bounding_box()

            # Run the space-oriented algorithm with primitive prim.
            vol = space_oriented_algorithm(
                [0, 0], num_x, num_y, num_z, threshold, minx, maxx, miny, maxy,
                minz, maxz, primitive=prim, test=True)
            vol_dif = abs(obj_vol - vol)

            # Print the results.
            print_test_results(prim, vol, obj_vol, vol_dif)

            clear_scene()

        print()


def main():
    """Execute the script."""
    time_start = time.time()

    # Test the different methods.
    test = False
    init(test)

    # Set info about threshold for reducing the bounding box.
    threshold = 0.05

    # Set number of primitives to divide the bounding box.
    num_x, num_y, num_z = 10, 10, 10
    print_num_primitives(num_x, num_y, num_z)

    if test:
        # Test all methods specified in the methods list.
        methods = ["tetra"]
        vol_test(num_x, num_y, num_z, threshold, methods=methods)
    else:
        # Determine bounding box and translate scene to origin.
        minx, maxx, miny, maxy, minz, maxz = bounding_box()
        difx, dify, difz = translate_to_origin(minx, maxx, miny, maxy, minz,
                                               maxz)

        # Update the new minimum and maximum coordinates.
        minx, maxx, miny, maxy, minz, maxz = update_minmax(
            minx, maxx, miny, maxy, minz, maxz, difx, dify, difz)

        print_bbox(minx, maxx, miny, maxy, minz, maxz)

        # Run the space-oriented algorithm with the assigned primitive.
        primitive = "tetra"
        volume = space_oriented_algorithm(
            ordered_meshes(), num_x, num_y, num_z, threshold, minx, maxx, miny,
            maxy, minz, maxz, primitive=primitive)

        print_total_volume(volume)

    print("Script Finished: %.4f sec" % (time.time() - time_start))


if __name__ == "__main__":
    main()

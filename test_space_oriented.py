import time
import sys
import bpy

sys.path.insert(1, "Documents/Scriptie/BachelorScriptieCode/")
from vol_generation_script import (init, clear_scene,
    space_oriented_algorithm, bounding_box)  # noqa: E402

D = bpy.data
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
ROUNDCUBE = bpy.ops.mesh.primitive_round_cube_add
GEM = bpy.ops.mesh.primitive_gem_add
DIAMOND = bpy.ops.mesh.primitive_diamond_add
SPONGE = bpy.ops.mesh.menger_sponge_add
PYRAMID = bpy.ops.mesh.primitive_steppyramid_add

TEST_VOLS = {
    "Suzanne": 2255372.6832, "Torus": 1174735.7603, "Cylinder": 6242890.2415,
    "Cone": 2080963.4138, "Cube": 8000000, "Sphere": 4121941.1484,
    "Icosphere": 3658712.0869, "Roundcube": 3927558.9273, "Gem": 1300599.1727,
    "Diamond": 1546155.8117, "Sponge": 5925925.82, "tk": 20238706.8905,
    "st": 734731.5605, "te": 513200.2638, "tp": 25297666.767,
    "dmesh": 1638237.8412, "Pyramid": 770000.0461}

LOCATIONS = [
    [(x, y, 2) for x in [6, 2, -2, -6] for y in [4, 0, -4]],
    [(x, y, -2) for x in [6, 2, -2, -6] for y in [4, 0, -4]]]


def print_primitive_size(length, width, height):
    """Print the size of the primitive used."""
    print("Primitive size: %.2f x %.2f x %.2f" % (length, width, height))


def print_test_results(name, prim, vol, obj_vol, vol_dif):
    """Print the test results of one particular object and method."""
    print("Object: %s, volume for space-oriented %s algorithm: %.4f ("
          "exact volume is %.4f)" % (name, prim, vol, obj_vol))
    print("The difference is thus %.4f or %.2f percent" %
          (vol_dif, 100 * vol_dif / vol))


def print_time(time_passed):
    """Print the amount of time passed during a test."""
    print("Time the test took to complete: %.4f sec." % time_passed)


def update_location(minx, maxx, miny, maxy, minz, maxz, loc):
    """Set bounding box at the right location."""
    minx += loc[0]
    maxx += loc[0]
    miny += loc[1]
    maxy += loc[1]
    minz += loc[2]
    maxz += loc[2]
    return minx, maxx, miny, maxy, minz, maxz


def translate_volume(prim, ind1, ind2):
    """Translate volume to correct location."""
    if prim == "tetra":
        base_name = "tetrahedron_volume"
    elif prim == "voxel":
        base_name = "cubic_volume"

    latest_obj = D.objects[base_name]
    latest_obj.location = LOCATIONS[ind1][ind2]
    latest_obj.name = base_name + str(ind1) + "-" + str(ind2)


def vol_test_space_oriented(object_list, length, width, height, threshold,
                            methods=["tetra"], draw_test=False, test=True):
    """Test the self-made algorithm against Blenders built-in for closed
    meshes. Volumes are here calculated in cm^3."""
    for j, prim in enumerate(methods):
        start_time = time.time()

        for i, (name, obj) in enumerate(object_list):
            # Create the object and get its volume.
            obj(location=LOCATIONS[j][i])
            obj_vol = TEST_VOLS[name]

            minx, maxx, miny, maxy, minz, maxz = bounding_box(name=name)

            # Run the space-oriented algorithm with primitive prim.
            if name == "dmesh":
                name = "dobj"
            ind = D.objects.keys().index(name)
            vol = space_oriented_algorithm(
                [ind, ind], length, width, height, threshold, minx, maxx, miny,
                maxy, minz, maxz, primitive=prim, draw=draw_test, test=test)
            translate_volume(prim, j, i)
            vol_dif = abs(obj_vol - vol)

            # Print the results.
            print_test_results(name, prim, vol, obj_vol, vol_dif)

            if not draw_test:
                clear_scene()

        time_passed = time.time() - start_time
        print_time(time_passed)

        print()


def main():
    """Test the space-oriented algorithm with Blender standard objects."""
    time_start = time.time()

    # Test the different methods.
    test = True
    draw_test = True
    init(test)

    # Set the test objects which are mesh primitives in Blender.
    object_list = [
        ("Suzanne", MONKEY), ("Torus", TORUS), ("Cylinder", CYL),
        ("Cone", CONE), ("Cube", CUBE), ("Sphere", UVSPHERE),
        ("Icosphere", ICOSPHERE), ("dmesh", B_DIAMOND),
        ("Roundcube", ROUNDCUBE), ("Diamond", DIAMOND),
        ("Sponge", SPONGE), ("Pyramid", PYRAMID)]

    # Set info about threshold (in cm) for reducing the bounding box.
    threshold = 5

    # Set number of primitives to divide the bounding box.
    # num_x, num_y, num_z = 50, 50, 50

    # Size in centimeters for the primitives.
    length, width, height = 10, 10, 10
    print_primitive_size(length, width, height)

    methods = ["voxel", "tetra"]
    vol_test_space_oriented(object_list, length, width, height, threshold,
                            methods=methods, draw_test=draw_test, test=test)

    print("Script Finished: %.4f sec." % (time.time() - time_start))


if __name__ == "__main__":
    main()

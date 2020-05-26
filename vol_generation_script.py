# This is a script made to calculate the volume between different surfaces
# of archaeological excavation grounds. It implements different methods to do
# so.
#
# Made by Bram Dekker (11428279) of the Univeristy of Amsterdam as a bachelor
# thesis.

# bpy, bmesh, mathutils and numpy are standard available in Blender's Python
# environment. Thus these modules dont have to be installed separately if the
# script is run inside Blender.
import bpy
import bmesh
import mathutils
import time
import os
import math
import numpy as np
import itertools
import sys
sys.setrecursionlimit(2500)

C = bpy.context
D = bpy.data

# Colors used to color the different layers. Feel free to add new RGBA colors
# to the RGBAS array or change them if you dont like certain colors.
RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
YELLOW = (1.0, 1.0, 0.0, 1.0)
ORANGE = (1.0, 0.5, 0.0, 1.0)
PURPLE = (0.5, 0.0, 1.0, 1.0)
PINK = (1.0, 0.4, 1.0, 1.0)
BROWN = (0.627, 0.322, 0.176, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
RGBAS = [RED, GREEN, BLUE, PINK, YELLOW, PURPLE, ORANGE, BROWN, BLACK, WHITE]


def init(test=False):
    """Initialize the scene by removing the cube, camera and light and import
    and orientate the trench objects."""
    if test:
        clear_scene()
    else:
        remove_volumes()
        rotate_objects()


def is_volume(obj):
    """Check if object is a generated volume or not."""
    if (obj.name.startswith("triangle_mesh_volume") or
            obj.name.startswith("cubic_volume") or
            obj.name.startswith("tetrahedron_volume")):
        return True

    return False


def remove_volumes():
    """Remove the generated volumes from the last script run."""
    vols = [obj for obj in D.objects if is_volume(obj)]
    for vol in vols:
        D.meshes.remove(vol.data)


def clear_scene():
    """Clear the scene from all objects."""
    if C.active_object is not None and C.active_object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    for m in D.meshes:
        D.meshes.remove(m)

    for obj in D.objects:
        D.objects.remove(obj, do_unlink=True)


def rotate_objects():
    """Rotate the objects back to their normal form, no rotation at all"""
    for name in D.objects:
        name.rotation_euler = (0.0, 0.0, 0.0)


def bounding_box(name=None):
    """Find the minimum and maximum x, y and z-coordinates."""
    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf
    if name:
        obj_list = [D.meshes[name]]
    else:
        obj_list = D.meshes

    for obj in obj_list:
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

    if (x_trans, y_trans, z_trans) == (0, 0, 0):
        return 0, 0, 0

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


def print_total_volume(volume):
    """Print the total volume."""
    print("The total volume is %.2f cm3 or %.2f m3." %
          (volume, volume / 1000000))


def print_primitive_size(length, width, height):
    """Print the size of the primitive used."""
    print("Primitive size: %.2f x %.2f x %.2f cm" % (length, width, height))


def print_layer_volume(test, volume, i):
    """Print the layer volume only if it is not a test run."""
    if not test:
        print("Layer %d has a volume of %.2f cm3 or %.2f m3" %
              (i, volume, volume / 1000000))


# Start of the space-oriented algorithm.
def ordered_meshes(selected_meshes):
    """Return the list of indices in de bpy.data.objects list of all
    objects from high to low (highest avg z-coordinate of vertices)."""
    mesh_list = []

    for obj in selected_meshes:
        cur_avg = np.average([vt.co[2] for vt in obj.data.vertices.values()])
        mesh_list.append((cur_avg, D.objects.values().index(obj)))
        obj.select_set(False)

    return [elem[1] for elem in sorted(mesh_list, reverse=True)]


def init_ply_file(file_path, file_name, num):
    """Initialize the ply file. This file will store all the centers of the
    primitives."""
    f = open(file_path + file_name, 'w')
    header = "ply\nformat ascii 1.0\nelement vertex {}\n".format(num) + \
             "property float x\n" + \
             "property float y\nproperty float z\nproperty uchar red\n" + \
             "property uchar green\nproperty uchar blue\n" + \
             "property int layer\nend_header\n"
    f.write(header)
    return f


def dist_btwn_pts(p1, p2):
    """Calculate the distance between 2 points in 3D space."""
    x_dif, y_dif, z_dif = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    return math.sqrt(x_dif**2 + y_dif**2 + z_dif**2)


def center_tetrahedron(v1, v2, v3, v4):
    """Return the center of a tetrahedron with vertices v1, v2, v3 and v4."""
    s = tuple(map(sum, zip(v1, v2, v3, v4)))
    return tuple(el / 4 for el in s)


def make_cuboid_verts(center, length, width, height):
    """Make a list of vertices for given center and sizes."""
    l, w, h = length / 2, width / 2, height / 2
    v0 = (center[0] - l, center[1] - w, center[2] - h)
    v1 = (center[0] + l, center[1] - w, center[2] - h)
    v2 = (center[0] - l, center[1] + w, center[2] - h)
    v3 = (center[0] + l, center[1] + w, center[2] - h)
    v4 = (center[0] - l, center[1] - w, center[2] + h)
    v5 = (center[0] + l, center[1] - w, center[2] + h)
    v6 = (center[0] - l, center[1] + w, center[2] + h)
    v7 = (center[0] + l, center[1] + w, center[2] + h)
    return [v0, v1, v2, v3, v4, v5, v6, v7]


def make_cuboid_vert_faces(verts):
    """Make a list of faces for given vertices of a cube."""
    f1 = (verts[0], verts[1], verts[3], verts[2])
    f2 = (verts[4], verts[5], verts[7], verts[6])
    f3 = (verts[0], verts[2], verts[6], verts[4])
    f4 = (verts[1], verts[3], verts[7], verts[5])
    f5 = (verts[0], verts[1], verts[5], verts[4])
    f6 = (verts[2], verts[3], verts[7], verts[6])
    return [f1, f2, f3, f4, f5, f6]


def get_other_tetra_verts(ratio_l, ratio_w, ratio_h, l, w, h, v0):
    """Get the 2 vertices of the tetrahedra which are not on the body
    diagonal."""
    vs = []

    # If center is 3/4 length or 1/4 length further than v0, we know 1 other
    # vertex for sure because of the way we divided the box.
    if ratio_l == 3:
        vs.append((v0[0] + l, v0[1], v0[2]))
    elif ratio_l == 1:
        vs.append((v0[0], v0[1] + w, v0[2] + h))

    # If center is 3/4 width or 1/4 width than v0, we know 1 other
    # vertex for sure because of the way we divided the box.
    if ratio_w == 3:
        vs.append((v0[0], v0[1] + w, v0[2]))
    elif ratio_w == 1:
        vs.append((v0[0] + l, v0[1], v0[2] + h))

    # If center is 3/4 height or 1/4 height further than v0, we know 1 other
    # vertex for sure because of the way we divided the box.
    if ratio_h == 3:
        vs.append((v0[0], v0[1], v0[2] + h))
    elif ratio_h == 1:
        vs.append((v0[0] + l, v0[1] + w, v0[2]))

    return vs[0], vs[1]


def make_tetra_verts(center, length, width, height, minx, miny, minz):
    """Make a list of vertices for given center and sizes."""
    # Get vertices closest to the (0, 0, 0) point or bottom diagonal corner.
    min_length = math.floor((center[0] - minx) / length) * length
    min_width = math.floor((center[1] - miny) / width) * width
    min_height = math.floor((center[2] - minz) / height) * height

    # Make the first 2 vertices which span the body diagonal of the box in
    # which the tetrahedron lies.
    v0 = (minx + min_length, miny + min_width, minz + min_height)
    v1 = (v0[0] + length, v0[1] + width, v0[2] + height)

    # Get the ratio between the center and v0.
    ratio_len = round(4 * (center[0] - v0[0]) / length)
    ratio_wid = round(4 * (center[1] - v0[1]) / width)
    ratio_hgt = round(4 * (center[2] - v0[2]) / height)

    # Get the other 2 vertices based on the difference between its center and
    # the bottom body diagonal vertex.
    v2, v3 = get_other_tetra_verts(
        ratio_len, ratio_wid, ratio_hgt, length, width, height, v0)
    return [v0, v1, v2, v3]


def make_tetra_vert_faces(verts):
    """Return the faces as tuples of BMVerts."""
    f1 = (verts[0], verts[1], verts[2])
    f2 = (verts[0], verts[1], verts[3])
    f3 = (verts[0], verts[2], verts[3])
    f4 = (verts[1], verts[2], verts[3])
    return [f1, f2, f3, f4]


def finish_bmesh(bm, index, primitive):
    """Write bmesh to a new mesh object and color it with the color on position
    index in the RGBAS list."""
    if primitive == "tetra":
        mesh = D.meshes.new("tetrahedron_volume")
    elif primitive == "cuboid":
        mesh = D.meshes.new("cubic_volume")

    # Create new mesh structure.
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate()
    mesh.update()

    # Create new object and give it data.
    if primitive == "tetra":
        new_object = D.objects.new("tetrahedron_volume", mesh)
    elif primitive == "cuboid":
        new_object = D.objects.new("cubic_volume", mesh)

    # Put object in the scene.
    C.collection.objects.link(new_object)
    C.view_layer.objects.active = new_object
    new_object.select_set(True)

    # Give the object the color rgba and deselect it.
    color = D.materials.new("Layer_color")
    C.active_object.data.materials.append(color)
    C.object.active_material.diffuse_color = RGBAS[index % len(RGBAS)]
    new_object.select_set(False)


def add_primitive(x, y, z, cur_bm, primitive, length, width, height, minx,
                  miny, minz):
    """Add a primitive with center (x, y, z) and a size of
    (length * width * height) to cur_bm mesh."""
    if primitive == "tetra":
        new_bm_verts = []

        # Get a list of the vertices of the tetrahedron with center (x, y, z).
        vs = make_tetra_verts((x, y, z), length, width, height, minx, miny,
                              minz)

        # Add the vertices to the mesh and store them in a list.
        for v in vs:
            new_vert = cur_bm.verts.new(v)
            new_bm_verts.append(new_vert)

        # Use the list of BMVerts to make the faces of the tetrahedron.
        fs = make_tetra_vert_faces(new_bm_verts)
        for f in fs:
            cur_bm.faces.new(f)
    elif primitive == "cuboid":
        new_bm_verts = []

        # Get a list of the vertices of the box with center (x, y, z).
        vs = make_cuboid_verts((x, y, z), length, width, height)

        # Add the vertices to the mesh and store them in a list.
        for v in vs:
            new_vert = cur_bm.verts.new(v)
            new_bm_verts.append(new_vert)

        # Use the list of BMVerts to make the faces of the tetrahedron.
        fs = make_cuboid_vert_faces(new_bm_verts)
        for f in fs:
            cur_bm.faces.new(f)


def draw_and_export(temp_file_name, primitive, length, width, height, minx,
                    miny, minz, num_primitives, draw, export, file_path,
                    ply_name):
    """Draw and export the calculated volume between the meshes if that is
    specified."""
    # If the layers need to be exported as PLY-file, initialize a PLY file
    # with a correct header.
    if export:
        ply_file = init_ply_file(file_path, ply_name, num_primitives)

    # If the layers need to be drawn, initialize a new mesh.
    if draw:
        cur_bm = bmesh.new()

    # If the layers dont need to be drawn or exported, return. Else
    if not draw and not export:
        return
    else:
        layer_index = 0
        with open(temp_file_name, 'r') as temp_file:
            for line in temp_file:
                # Initialize a PLY-file if needed.
                if export:
                    ply_file.write(line)

                # Extract the center and layer index of the next primitive, if
                # the mesh has to be drawn.
                if draw:
                    x, y, z, _, _, _, i = line.rstrip().split(' ')
                    x, y, z = float(x), float(y), float(z)
                    i = int(i)

                    # Draw the layer to the screen if the last primitive of
                    # this layer is found. Pass index to specify color.
                    if not i == layer_index:
                        finish_bmesh(cur_bm, (i - 1), primitive)
                        cur_bm = bmesh.new()
                        layer_index = i

                    # Add the primitive to the current mesh.
                    add_primitive(x, y, z, cur_bm, primitive, length, width,
                                  height, minx, miny, minz)

            # Draw the last layer.
            if draw:
                finish_bmesh(cur_bm, layer_index, primitive)
    if export:
        ply_file.close()


def set_vol_primitive(length, width, height, primitive):
    """Calculate the volume of the specified primitive in cubic centimetres."""
    if primitive == "cuboid":
        vol_primitive = length * width * height * 1000000
    elif primitive == "tetra":
        vol_primitive = 1000000 * (length * width * height) / 6

    return vol_primitive


def check_inside_mesh(mesh, center, max_distance, direc="up"):
    """Check if center lays inside closed mesh or not."""
    hit = True
    hit_count = 0
    cast_point = center

    # Set appropriate ray direction and adder.
    if direc == "up":
        adder = (0, 0, 0.001)
        ray_direction = (0, 0, 1)
    elif direc == "down":
        adder = (0, 0, -0.001)
        ray_direction = (0, 0, -1)

    # Cast rays as long as it hits the mesh. Count the amount of times the ray
    # hits the mesh.
    while hit:
        cast_result = D.objects[mesh].ray_cast(
            cast_point, ray_direction, distance=max_distance)

        if cast_result[0]:
            hit_count += 1
            cast_point = tuple(sum(x) for x in zip(cast_result[1], adder))
        else:
            hit = False

    # If there are an uneven amount of hits, center is inside the closed
    # mesh and thus True is returned.
    return (hit_count % 2) == 1


def decide_in_volume(upper_mesh, lower_mesh, center, max_distance, threshold):
    """Decide if the primitive on the given position belongs to the volume
    between the meshes."""
    # Cast ray up to upper mesh and check if it is in between the 2 meshes.
    res_up = D.objects[upper_mesh].ray_cast(
        center, (0, 0, 1), distance=max_distance)
    if (not res_up[0] or not check_inside_mesh(
            upper_mesh, center, max_distance, direc="up")):
        return False

    # Cast ray down to lower mesh and check if it is in between the 2 meshes.
    res_down = D.objects[lower_mesh].ray_cast(
        center, (0, 0, -1), distance=max_distance)
    if (not res_down[0] or not check_inside_mesh(
            lower_mesh, center, max_distance, direc="down")):
        return False

    # If center is in between the 2 meshes, check if its exceed the threshold.
    if dist_btwn_pts(res_up[1], res_down[1]) < threshold:
        return False

    return True


def make_point_list(minx, maxx, miny, maxy, minz, maxz, l, w, h):
    """Make a list of cartesian points between boundaries."""
    # Make lists for x, y and z-coordinates that divide up the bounding box.
    xs = np.arange(minx, maxx + l, l)
    ys = np.arange(miny, maxy + w, w)
    zs = np.arange(minz, maxz + h, h)

    # Combine the coordinates to 3D Cartesian coordinates.
    return ((x, y, z) for x in xs for y in ys for z in zs)


def process_cuboid(
        x, y, z, l, w, h, upper_mesh, lower_mesh, max_distance,
        vol_primitive, threshold, f, layer_index,
        r, g, b):
    """Determine if cuboid is part of the volume or not."""
    # Initialize volume to zero.
    volume = 0

    # Determine center of cuboid.
    center = (x + (l / 2), y + (w / 2), z + (h / 2))

    # Decide if cuboid is part of the volume.
    in_volume = decide_in_volume(
        upper_mesh, lower_mesh, center, max_distance, threshold)

    # If the cuboid is part of the volume, add it to the primitive list and set
    # the volume.
    if in_volume:
        line = "{} {} {} {} {} {} {}\n".format(
            center[0], center[1], center[2], r, g, b, layer_index)
        f.write(line)
        volume += vol_primitive

    return volume


def process_tetra(
        x, y, z, l, w, h, upper_mesh, lower_mesh, max_distance,
        vol_primitive, threshold, f, layer_index,
        r, g, b):
    """Determine if tetra is part of the volume or not."""
    # Initialize volume to zero.
    volume = 0

    # Get all vertices for the 6 tetrahedra.
    v1, v2 = (x, y, z), (x + l, y + w, z + h)
    other_vs = [(x+l, y, z), (x+l, y+w, z), (x, y+w, z), (x, y+w, z+h),
                (x, y, z+h), (x+l, y, z+h), (x+l, y, z)]
    for v3, v4 in zip(other_vs[:-1], other_vs[1:]):
        # Determine center of tetrahedron.
        center = center_tetrahedron(v1, v2, v3, v4)

        # Decide if the tetrahedron is part of the volume.
        in_volume = decide_in_volume(
            upper_mesh, lower_mesh, center, max_distance, threshold)

        # If the tetrahedron is part of the volume, add it to the primitive
        # list and add its volume to the volume variable.
        if in_volume:
            line = "{} {} {} {} {} {} {}\n".format(
                center[0], center[1], center[2], r, g, b, layer_index)
            f.write(line)
            volume += vol_primitive

    return volume


def space_oriented_volume_between_meshes(
        length, width, height, threshold, minx, maxx, miny, maxy, minz, maxz,
        upper_mesh, lower_mesh, max_distance, primitive, f, layer_index,
        layer_color):
    """Calculate the volume and make a 3D model of the volume between 2
    meshes."""
    # Initialize volume primitives list and the volume.
    r, g, b = [math.ceil(255 * color) for color in layer_color[:3]]
    cur_volume = 0

    # Determine primitive size
    l, w, h = length, width, height

    # Set volume of the primitive used.
    vol_primitive = set_vol_primitive(l, w, h, primitive)

    # Go over every primitive in bounding box space
    points = make_point_list(minx, maxx, miny, maxy, minz, maxz, l, w, h)

    for x, y, z in points:
        if primitive == "cuboid":
            # Check if current cuboid is part of the volume.
            cur_volume += process_cuboid(
                x, y, z, l, w, h, upper_mesh, lower_mesh, max_distance,
                vol_primitive, threshold, f, layer_index,
                r, g, b)
        elif primitive == "tetra":
            # Check which tetrahedra in the current cuboid are part of the
            # volume.
            cur_volume += process_tetra(
                x, y, z, l, w, h, upper_mesh, lower_mesh, max_distance,
                vol_primitive, threshold, f, layer_index,
                r, g, b)

    return cur_volume


def space_oriented_algorithm(
        meshes, length, width, height, threshold, minx, maxx, miny, maxy, minz,
        maxz, primitive="cuboid", draw=True, test=False, export=False,
        file_path="Downloads/", ply_name="pointmesh.ply"):
    """Space-oriented algorithm to make a 3D model out of multiple trianglar
    meshes."""
    # Initialize the total volume and maximum vertical distance.
    temp_file_name = "vol_gen_temp_storage.txt"
    f = open(temp_file_name, 'w')
    tot_volume = 0
    max_distance = maxz - minz
    length, width, height = length / 100, width / 100, height / 100
    threshold /= 100

    # For all ordered mesh pairs from top to bottom, run the space-oriented
    # algorithm to get volume and list of primitives.
    for i, (upper_mesh, lower_mesh) in enumerate(zip(meshes[:-1], meshes[1:])):
        layer_color = RGBAS[lower_mesh % len(RGBAS)]

        # Execute space-oriented algorithm between 2 meshes.
        volume = space_oriented_volume_between_meshes(
            length, width, height, threshold, minx, maxx, miny, maxy, minz,
            maxz, upper_mesh, lower_mesh, max_distance, primitive, f, i,
            layer_color)

        # Print the size of the primitives used.
        print_layer_volume(test, volume, i)

        # Update total volume.
        tot_volume += volume

    # Close the temporary file.
    f.close()

    # Calculate the number of primitives added which is needed for the
    # PLY-file.
    number_of_primitives = round(
        tot_volume / set_vol_primitive(length, width, height, primitive))

    # Draw the layers in Blender and export all as 1 PLY-file if specified.
    draw_and_export(temp_file_name, primitive, length, width, height, minx,
                    miny, minz, number_of_primitives, draw, export, file_path,
                    ply_name)

    # Remove the temporary file.
    os.remove(temp_file_name)

    return tot_volume


# Start of the object-oriented algortihm.
def find_boundary_vertices(bm):
    """Return all the vertices that are in the boundary of the object."""
    return [v for v in bm.verts if v.is_valid and v.is_boundary]


def find_boundary_edges(bm):
    """Return all the edges that are in the boundary of the object."""
    return [e for e in bm.edges if e.is_valid and e.is_boundary]


def dist_btwn_2d_pts(p1, p2):
    """Calculate the distance between 2 2D points."""
    x_dif, y_dif = p1[0] - p2[0], p1[1] - p2[1]
    return math.sqrt(x_dif**2 + y_dif**2)


def calc_centroid_island(island):
    """Calculate the centroid of a island by taking average values for x, y and
    z coordinates."""
    coordinates = [v.co for v in island]
    centroid = tuple(map(lambda x: sum(x) / float(len(x)), zip(*coordinates)))
    return centroid


def find_closest_island(centroid, cen_list, islands, index):
    """Find element in cen_list that is closest to centroid."""
    # Initiliaze the minimum distance, minimum index in islands list and the
    # minimum center of a island.
    min_dist = math.inf
    min_index = 0
    min_cen = (0, 0, 0)

    # For every center of an island that has not yet been found, calculate its
    # distance in 3D space to the specific island. If this distance is smaller
    # than the current minimum distance, set it to minimum.
    for (i, c) in cen_list:
        dist = dist_btwn_2d_pts(centroid[:2], c[:2])
        if dist < min_dist:
            min_dist = dist
            min_index = i
            min_cen = c

    return min_index, min_cen, min_dist


def find_mesh_pairs(islands, upper_verts):
    """Find pairs of vertex rings that need to be stiched together."""
    # If there are an uneven number of islands, pop the last and thus
    # smallest island.
    if len(islands) % 2 != 0:
        islands.pop()

    # Make a list with tuples of indices and centroids of the islands.
    centroids = []
    for i, isle in enumerate(islands):
        if isle[0] in upper_verts:
            upper = True
        else:
            upper = False

        centroids.append((i, calc_centroid_island(isle), upper))

    # Search for pairs beginning with the largest islands. The pairs are
    # based on centroid distance.
    pairs, found = [], []

    for i, c, u in centroids:
        if (i, c) not in found:
            not_yet_found = [cen[:2] for cen in centroids if cen[-1] != u and
                             cen not in found]
            j, cen, dist = find_closest_island(c, not_yet_found, islands, i)
            if dist > 0.25:
                continue
            else:
                found.append((i, c))
                found.append((j, cen))
                pairs.append([islands[i], islands[j]])

    return pairs


def remove_loose_parts(bm):
    """Remove loose vertices and edges from the mesh."""
    # Remove faces that are not connected to any other face, e.g. all its
    # edges are boundary edges.
    for f in bm.faces:
        if all([e.is_boundary for e in f.edges]):
            bm.faces.remove(f)

    # Remove vertices that have no face connected to it.
    loose_verts = [v for v in bm.verts if v.is_wire]
    for v in loose_verts:
        bm.verts.remove(v)

    # Remove edges that have no face conncectd to it.
    loose_edges = [e for e in bm.edges if e.is_wire]
    for e in loose_edges:
        bm.edges.remove(e)


# This is a function written by batFINGER and can be found here:
# https://blender.stackexchange.com/questions/75332/how-to-find-the-number-of-loose-parts-with-blenders-python-api
def walk_island(vert):
    """Walk all un-tagged linked verts."""
    vert.tag = True
    yield(vert)
    linked_verts = [e.other_vert(vert) for e in vert.link_edges
                    if not e.other_vert(vert).tag and e.is_boundary]

    for v in linked_verts:
        if v.tag:
            continue
        yield from walk_island(v)


# This is a function written by batFINGER and can be found here:
# https://blender.stackexchange.com/questions/75332/how-to-find-the-number-of-loose-parts-with-blenders-python-api
def get_islands(bm, verts=[]):
    """Get islands of vertices given all vertices."""
    def tag(verts, switch):
        for v in verts:
            v.tag = switch
    tag(bm.verts, True)
    tag(verts, False)
    ret = {"islands": []}
    verts = set(verts)
    while verts:
        v = verts.pop()
        verts.add(v)
        island = set(walk_island(v))
        ret["islands"].append(list(island))
        tag(island, False)  # remove tag = True
        verts -= island
    return ret


def make_next_verts(path1, path2, edges, good_verts):
    """Get the next vertices for path1 and path2."""
    # Get the vertices to which the last path vertex is connected to.
    verts1 = [e.other_vert(path1[-1]) for e in edges
              if path1[-1] in e.verts]
    verts2 = [e.other_vert(path2[-1]) for e in edges
              if path2[-1] in e.verts]

    # Get the vertex that is in good_verts and not in the path and is
    # connected to the last path vertex.
    next_vert1 = [v for v in verts1 if v in good_verts and v not in path1]
    next_vert2 = [v for v in verts2 if v in good_verts and v not in path2]

    return next_vert1, next_vert2


def correct_next_verts(next_vert1, next_vert2, wrong_verts, path1, path2):
    """Returns True if both next_verts are correct."""
    # If the next verts do not not exists or are wrong vertices, they are not
    # correct. Else they are correct.
    if not next_vert1 or not next_vert2:
        return False
    elif next_vert1[0] == path1[0] or next_vert2[0] == path1[0]:
        return False
    elif next_vert1[0] in wrong_verts or next_vert2[0] in wrong_verts:
        return False

    return True


def check_for_overlap(path1, path2):
    """Check if the 2 paths overlap."""
    # The paths overlap if their last vertices are the same or if the last
    # vertex of a path is the same as the second-last of the other path.
    overlap = False
    if path1[-1] == path2[-1]:
        overlap = True
        path1.pop()
    elif path1[-1] == path2[-2] or path2[-1] == path1[-2]:
        overlap = True
        path1.pop()
        path2.pop()

    return overlap


def small_loop_between(start, e1, e2, edges, wrong_verts, good_verts, count):
    """Determines if there exists a small loop between e1 and e2."""
    # Initialize 2 paths from the start vertex with 4 boundary edges connected
    # to it.
    path1 = [start, e1.other_vert(start)]
    path2 = [start, e2.other_vert(start)]

    # Initialize the loop_found and verts that are in the loop variables.
    loop_found = False
    verts = []

    # Search for a loop while it is not found and count is not less than 0.
    while not loop_found:
        count -= 1
        if count < 0:
            break

        # Get the next vertices in both paths.
        next_vert1, next_vert2 = make_next_verts(path1, path2, edges,
                                                 good_verts)

        # If there is no next vertex or it is a wrong vertex, stop the search.
        if not correct_next_verts(next_vert1, next_vert2, wrong_verts,
                                  path1, path2):
            break

        # Append the next vertices to their paths.
        path1.append(next_vert1[0])
        path2.append(next_vert2[0])

        # Check if a loop is found, e.g. are there overlapping elements in the
        # paths. If so, pop paths accordingly and set loop_found to true.
        loop_found = check_for_overlap(path1, path2)

    # Put all the vertices in the loop in a list, without the start vertex.
    if loop_found:
        verts = path1 + path2
        verts = [v for v in verts if v != start]

    return loop_found, verts


def find_double_loop(start, edges, still_wrong_verts):
    """Find a loop that is between 2 vertices."""
    start_edges = [e for e in edges if start in e.verts]
    paths = []

    # Make all 4 paths from the 4 different edges to a vertex in wrong_verts.
    for e in start_edges:
        found_path = False
        p = [start, e.other_vert(start)]

        while not found_path:
            next_vert = [e.other_vert(p[-1]) for e in edges
                         if p[-1] in e.verts and not p[-2] in e.verts]
            p.append(next_vert[0])
            if p[-1] in still_wrong_verts:
                found_path = True
                paths.append(p)

    # Get the target vertex and the 2 shortest paths toward it.
    path_ends = [p[-1] for p in paths]
    other_v = list(set([v for v in path_ends if path_ends.count(v) > 1]))[0]
    end_edges = [e for e in edges if other_v in e.verts]
    paths_to_other_v = [p for p in paths if p[-1] == other_v]
    paths_to_other_v.sort(key=len)
    shortest_paths = paths_to_other_v[:2]

    # Get loop vertices and loop edges and return them.
    loop_verts = [v for p in shortest_paths for v in p
                  if v not in still_wrong_verts]
    loop_edges = [e for e in start_edges if e.other_vert(start) in loop_verts]
    loop_edges += [e for e in end_edges if e.other_vert(other_v) in loop_verts]
    return loop_verts, loop_edges, other_v


def find_and_remove_double_loops(still_wrong_verts, b_edges,
                                 edges_to_be_removed, verts_to_be_removed):
    """Find and remove any double loops that are still present in the mesh."""
    still_wrong_found = []
    for v in still_wrong_verts:
        if v not in still_wrong_found:
            still_wrong_found.append(v)
            loop_verts, loop_edges, other_v = find_double_loop(
                v, b_edges, still_wrong_verts)
            edges_to_be_removed.extend(loop_edges)
            verts_to_be_removed.extend(loop_verts)
            still_wrong_found.append(other_v)


def remove_loops(mesh, b_edges, edges_to_be_removed, verts_to_be_removed,
                 wrong_verts, combinations):
    """Remove small loops from a vertex island."""
    # Initialize incrementer to 0 and vertices found list to empty.
    verts_found = []
    count = 1

    # While there are vertices with another number than 2 connected edges,
    # try to find and remove the smaller loop on these vertices.
    while len(wrong_verts) > len(verts_found):
        # Loop over wrong vertices where the loop has not yet been found.
        for v in [v for v in wrong_verts if v not in verts_found]:

            # For all edge pair combinations, try to find loop between them.
            edges_list = [e for e in b_edges if v in e.verts]
            for ind1, ind2 in combinations:
                edge1, edge2 = edges_list[ind1], edges_list[ind2]
                loop_found, loop_verts = small_loop_between(
                    v, edge1, edge2, b_edges,
                    [v for v in wrong_verts if v not in verts_found],
                    mesh, count)

                # If a loop is found, add edges and vertices in the loop to the
                # to_be_removed lists. Also remove the vertices from the
                # current mesh and add the start vertex to verts_found.
                if loop_found:
                    edges_to_be_removed.extend([edge1, edge2])
                    verts_to_be_removed.extend(loop_verts)
                    for vert in loop_verts:
                        if vert in mesh:
                            mesh.remove(vert)
                    verts_found.append(v)
                    break

        count *= 2

        # If the incrementer exceeds the length of counts, break. Remaining
        # loops are too big to be found -> increase counts if necessary.
        if count > 150:
            break

    # Find double loops if there are still wrong vertices that have not been
    # removed.
    still_wrong_verts = [v for v in wrong_verts if v not in verts_found]
    find_and_remove_double_loops(
        still_wrong_verts, b_edges, edges_to_be_removed, verts_to_be_removed)


def remove_small_loops(upper_mesh, lower_mesh, b_edges):
    """Remove small loops from the boundary vertices, e.g. vertices with 4
    edges."""
    # Initialize edges and vertices that need to be removed lists.
    edges_to_be_removed, verts_to_be_removed = [], []

    # Determine all wrong vertices, e.g. vertices with an other number than 2
    # boundary vertices connected to it.
    upper_wrong_verts = [v for v in upper_mesh
                         if len([e for e in b_edges if v in e.verts]) != 2]
    lower_wrong_verts = [v for v in lower_mesh
                         if len([e for e in b_edges if v in e.verts]) != 2]

    # Make combinations of the edges, based on 4 edges.
    combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Remove loops in the upper mesh.
    remove_loops(
        upper_mesh, b_edges, edges_to_be_removed, verts_to_be_removed,
        upper_wrong_verts, combinations)

    # Remove loops in the lower mesh.
    remove_loops(
        lower_mesh, b_edges, edges_to_be_removed, verts_to_be_removed,
        lower_wrong_verts, combinations)

    return upper_mesh, lower_mesh, edges_to_be_removed, verts_to_be_removed


def sort_vert_list(sorted_list, unsorted, boundary_edges):
    """Create a sorted list of vertices based on edge connections of boundary
    edges."""
    # Add vertices while the length of the sorted list is less than the
    # original list.
    while len(sorted_list) < len(unsorted):
        # Get the last vertex of the list and get the vertices to which this
        # vertex is connected via the connected edges.
        cur_vert = sorted_list[-1]
        edges = [e for e in boundary_edges if cur_vert in e.verts]
        connected = [e.other_vert(cur_vert) for e in edges]

        # If the length of the sorted is 1, just add the first connected
        # vertex. Else add the one not yet in the sorted list.
        if len(sorted_list) == 1:
            sorted_list.append(connected[0])
        else:
            next_vert = [v for v in connected if v != sorted_list[-2]]
            sorted_list.extend(next_vert)


def sort_vert_lists(upper_mesh, lower_mesh, b_edges):
    """Sort the lists of BMVerts, so that vertices that are physically next to
    each other, are also next to each other in the list."""
    # Initialize the lists with the vertex at index 0.
    up, low = [upper_mesh[0]], [lower_mesh[0]]

    # Sort both island vertices.
    sort_vert_list(up, upper_mesh, b_edges)
    sort_vert_list(low, lower_mesh, b_edges)

    return up, low


def find_long_mesh(mesh1, mesh2):
    """Return longest mesh as first element."""
    if len(mesh1) >= len(mesh2):
        return mesh1, mesh2
    return mesh2, mesh1


def horizontal_edge(vector, rad):
    """Check if the edge is too horinzontal."""
    up = mathutils.Vector((0, 0, 1))
    down = mathutils.Vector((0, 0, -1))

    # Edge must be upwards or downwards with an angle <0.6 radians.
    if abs(up.angle(vector)) > rad and abs(down.angle(vector)) > rad:
        return True

    return False


def find_min_edge_to_short_mesh(short_mesh, v):
    """Find the shortest edge to the short mesh, with priority for
    vertical edges."""
    edge_found, hor_edge_found = False, False
    rad = 0.6
    min_dist, min_dist_hor = math.inf, math.inf

    while not edge_found:
        for j, other_v in enumerate(short_mesh):
            cur_vec = other_v.co - v.co

            cur_dist = dist_btwn_pts(v.co, other_v.co)

            # If the edge is too horizontal dont use it directly.
            if horizontal_edge(cur_vec, rad):
                hor_edge_found = True

                if cur_dist < min_dist_hor:
                    min_dist_hor = cur_dist
                    min_edge_hor = j

                continue

            # If edge is not close to perpendicular to its base edges,
            # dont use it.
            edge_found = True
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_edge = j

        # Use horizontal edge if it is way shorter than the vertical one.
        if edge_found:
            if hor_edge_found and (3 * min_dist_hor) < min_dist:
                min_edge = min_edge_hor

        rad += 0.2

    return min_edge


def jasnom_step_1_and_2(long_mesh, short_mesh):
    """Execute step 1 and step 2 of the JASNOM algorithm between the long mesh
    and the short mesh."""
    first_edges, second_edges = [], []

    # Loop over longest mesh vertices and make edge to closest vertex on the
    # shorter mesh.
    for i, v in enumerate(long_mesh):
        min_edge = find_min_edge_to_short_mesh(short_mesh, v)
        first_edges.append((i, min_edge))

    # Loop in reversed order over longer mesh and make a second edge to the
    # shorter mesh to the target vertex of the previous vertex in the longer
    # mesh.
    for i, v in reversed(first_edges):
        new_index = (i + 1) % len(first_edges)
        second_edges.append((new_index, v))

    all_edges = list(set(first_edges + second_edges))
    return all_edges


def make_and_stitch_unconnected_vert_groups(
        all_edges, long_mesh, short_mesh, not_connected,
        connected_in_this_step):
    """Mke v groups/"""
    still_not_connected = [v for v in not_connected
                           if v not in connected_in_this_step]
    v_index_groups = []
    for v in still_not_connected:
        short_mesh[v].select = True
        added = False
        prev_vert = (v - 1) % len(short_mesh)
        next_vert = (v + 1) % len(short_mesh)

        for group in v_index_groups:
            if prev_vert in group or next_vert in group:
                group.append(v)
                added = True

        if not added:
            v_index_groups.append([v])

    v_groups = [[short_mesh[i] for i in indices] for indices in v_index_groups]
    avg_location = [calc_centroid_island(group) for group in v_groups]

    for i, loc in enumerate(avg_location):
        min_dist = math.inf

        for j, other_v in enumerate(long_mesh):
            cur_dist = dist_btwn_pts(loc, other_v.co)
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_edge = j

        for v_index in v_index_groups[i]:
            all_edges.append((min_edge, v_index))


def jasnom_step_3(long_mesh, short_mesh, all_edges):
    """Execute step 3 of the JASNOM algorithm between the long mesh and the
    short mesh."""
    # Check if at least 1 edge connects a vertex of the shorter mesh to the
    # longer mesh. If not, make an edge to the target vertex of a neighbour
    # vertex in the shorter mesh.
    not_connected = [i for i in range(len(short_mesh))
                     if i not in [s for _, s in all_edges]]
    connected_in_this_step = []

    for v_index in not_connected:
        prev_index = ((v_index - 1) % len(short_mesh))
        prev_with_edge = [v1 for (v1, v2) in all_edges if v2 == prev_index]
        while not prev_with_edge:
            prev_index = (prev_index - 1) % len(short_mesh)
            prev_with_edge = [v1 for (v1, v2) in all_edges if v2 == prev_index]

        next_index = ((v_index + 1) % len(short_mesh))
        next_with_edge = [v1 for (v1, v2) in all_edges if v2 == next_index]
        while not next_with_edge:
            next_index = (next_index + 1) % len(short_mesh)
            next_with_edge = [v1 for (v1, v2) in all_edges if v2 == next_index]

        common_vertices = list(set(prev_with_edge) & set(next_with_edge))
        if common_vertices:
            target_vertex = common_vertices[0]
            all_edges.append((target_vertex, v_index))
            connected_in_this_step.append(v_index)

    # If there are vertices still not connected, make them into groups which
    # consists of vertices next to each other and add them all to the vertex
    # in the long mesh closest to the average location.
    make_and_stitch_unconnected_vert_groups(
        all_edges, long_mesh, short_mesh, not_connected,
        connected_in_this_step)


def make_jasnom_faces(long_mesh, short_mesh, new_edges, b_edges):
    """Make faces that are generated by the JASNOM algorithm."""
    new_faces = []
    vertex_pairs = [(e.verts[0], e.verts[1]) for e in b_edges]

    # For all vertices in the long mesh, check if it has multiple edges to the
    # short mesh. If so, make the faces for these edges.
    for v in long_mesh:
        edges_with_v = [e for e in new_edges if v == e[0]]
        if len(edges_with_v) >= 2:
            verts = [e[1] for e in edges_with_v]
            edges = [(v1, v2) for v1, v2 in itertools.combinations(verts, 2)
                     if (v1, v2) in vertex_pairs or (v2, v1) in vertex_pairs]

            for v1, v2 in edges:
                new_faces.append((v, v1, v2))

    # For all vertices in the short mesh, check if it has multiple edges to the
    # long mesh. If so, make the faces for these edges.
    for v in short_mesh:
        edges_with_v = [e for e in new_edges if v == e[1]]
        if len(edges_with_v) >= 2:
            verts = [e[0] for e in edges_with_v]
            edges = [(v1, v2) for v1, v2 in itertools.combinations(verts, 2)
                     if (v1, v2) in vertex_pairs or (v2, v1) in vertex_pairs]

            for v1, v2 in edges:
                new_faces.append((v, v1, v2))

    return new_faces


def jasnom_stitch(upper_mesh, lower_mesh, b_edges):
    """Stitch 2 vertex rings together using the JASNOM algorithm. The JASNOM
    algorithm ensures that there are no edges crossing each other."""
    long_mesh, short_mesh = find_long_mesh(upper_mesh, lower_mesh)

    all_edges = jasnom_step_1_and_2(long_mesh, short_mesh)
    jasnom_step_3(long_mesh, short_mesh, all_edges)

    # Make lists of edges and faces that need to be added to the mesh.
    new_edges = [(long_mesh[i], short_mesh[j]) for (i, j) in all_edges]
    new_faces = make_jasnom_faces(long_mesh, short_mesh, new_edges, b_edges)

    return new_edges, new_faces


def remove_loops_and_stitch(mesh1, mesh2, b_edges, bmsh, new_edges,
                            new_faces, i):
    """Remove small loops and stitch island pair together."""
    # Remove small loops in the island vertices.
    mesh1, mesh2, edges_removed, verts_removed = remove_small_loops(
        mesh1, mesh2, b_edges)

    # Remove the edges and vertices that are in small loops.
    for e in edges_removed:
        if e.is_valid:
            bmsh.edges.remove(e)
    for v in verts_removed:
        if v.is_valid:
            bmsh.verts.remove(v)

    # Update the boundary edges according to the removal.
    b_edges = [e for e in b_edges if e.is_valid]

    # Sort the vertex islands so that vertices next to each other in the
    # list are also next to each other in the island.
    mesh1, mesh2 = sort_vert_lists(mesh1, mesh2, b_edges)

    # Stitch the 2 islands together using the JASNOM algorithm and add the
    # new edges and faces to the list.
    edge_list, face_list = jasnom_stitch(mesh1, mesh2, b_edges)
    new_edges.extend(edge_list)
    new_faces.extend(face_list)

    return mesh1, mesh2, b_edges


def remove_small_islands(obj):
    """Remove small islands from the mesh."""
    for ob in D.objects:
        ob.select_set(False)

    # Select only the specified object and separate it into loose parts.
    obj.select_set(True)
    bpy.ops.mesh.separate(type="LOOSE")

    # Remove all objects which are too small.
    for o in C.selected_objects:
        if len(o.data.vertices) < 1000:
            D.meshes.remove(o.data)

    # Join the remaining big objects together.
    C.view_layer.objects.active = C.selected_objects[0]
    if len(C.selected_objects) > 1:
        bpy.ops.object.join()

    # Remove the mesh data that is not used.
    for m in D.meshes:
        if m.users == 0:
            D.meshes.remove(m)

    return C.selected_objects[0]


def stitch_meshes(bm, upper_verts):
    """Stitch the received meshes together using a 3D version of JANSNOM."""
    # Remove loose edges and vertices.
    remove_loose_parts(bm)

    # Determince boundary vertices and edges.
    b_verts = find_boundary_vertices(bm)
    b_edges = find_boundary_edges(bm)

    # Calculate vertex islands (loose parts) based on the boundary vertices and
    # sort them from large to small.
    islands = [island for island in get_islands(bm, verts=b_verts)["islands"]]
    islands.sort(key=len, reverse=True)

    # Create pairs of island vertices that are close together.
    island_pairs = find_mesh_pairs(islands, upper_verts)

    # Stitch the pairs together using the JASNOM algorithm.
    new_edges, new_faces = [], []
    for i, [upper_mesh, lower_mesh] in enumerate(island_pairs):
        upper_mesh, lower_mesh, b_edges = remove_loops_and_stitch(
            upper_mesh, lower_mesh, b_edges, bm, new_edges, new_faces, i)

    # Add new edges and faces to the mesh.
    for edge in new_edges:
        if bm.edges.get(edge) is None:
            bm.edges.new(edge)

    for face in new_faces:
        if bm.faces.get(face) is None:
            bm.faces.new(face)

    return bm


def fill_up_islands(bm, vert_islands):
    """Fill up all the holes that are still in the mesh."""
    for isle in vert_islands:
        if len(isle) > 2:
            face_edge_dict = bmesh.ops.contextual_create(
                bm, geom=isle, use_smooth=False)

            bmesh.ops.triangulate(bm, faces=face_edge_dict["faces"])


def volume_tetra_with_origin(v1, v2, v3):
    """Calculate the volume of the tetrahedron with vertices v1, v2 and v3
    connected to the origin."""
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    x3, y3, z3 = v3
    return (-x3 * y2 * z1 + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 -
            x2 * y1 * z3 + x1 * y2 * z3) / 6


def make_good_center(o):
    """Determine the center of the object and if it does not lies inside the
    object, shift it so that it is."""
    # Determine the center by taking the average of all vertices.
    center = tuple(np.average([v.co for v in o.data.vertices], axis=0))

    # Check if the center is actually inside the object and if not shift it
    # in the appropriate direction.
    if not check_inside_mesh(D.objects.values().index(o), center, 100):
        down_point = o.ray_cast(center, (0, 0, -1), distance=100)
        up_point = o.ray_cast(center, (0, 0, 1), distance=100)
        shift = mathutils.Vector((0, 0, 0.001))

        if down_point[0]:
            sec_down_point = o.ray_cast(down_point[1] - shift,
                                        (0, 0, -1), distance=100)
            center = (down_point[1] + sec_down_point[1]) / 2
        elif up_point[0]:
            sec_up_point = o.ray_cast(up_point[1] + shift,
                                      (0, 0, 1), distance=100)
            center = (up_point[1] + sec_up_point[1]) / 2

    return center


def process_loose_object(o):
    """Process the losse object o by filling remaing holes and calculate its
    volume."""
    # Determine the center.
    center = make_good_center(o)

    # Make a new bmesh object.
    temp_mesh = bmesh.new()
    temp_mesh.from_mesh(o.data)

    # Remove any loose faces, edges and vertices.
    remove_loose_parts(temp_mesh)

    # Fill up the remaining islands/holes.
    m_verts = [v for v in temp_mesh.verts if v.is_boundary]
    islands = [island for island in get_islands(
        temp_mesh, verts=m_verts)["islands"]]
    fill_up_islands(temp_mesh, islands)

    # Update the normals of the faces.
    temp_mesh.normal_update()
    good_faces = [f for f in temp_mesh.faces
                  if all([e.is_manifold for e in f.edges])]
    bmesh.ops.recalc_face_normals(temp_mesh, faces=good_faces)

    # Calculate the volume of the object.
    cur_vol = abs(sum([volume_tetra_with_origin(
        *[tuple(np.subtract(v.co, center)) for v in f.verts])
            for f in good_faces]))

    # Write bmesh back to mesh object and free it.
    temp_mesh.to_mesh(o.data)
    temp_mesh.free()

    return cur_vol


def fill_holes_and_calc_volume(obj):
    """Remove small islands, fill holes up and calculate volume for every
    loose part."""
    volume = 0

    for ob in D.objects:
        ob.select_set(False)

    # Select only the specified object and separate it into loose parts.
    obj.select_set(True)
    bpy.ops.mesh.separate(type="LOOSE")

    # Remove all objects which are too small.
    for o in C.selected_objects:
        if len(o.data.vertices) < 1000:
            D.meshes.remove(o.data)
        else:
            volume += process_loose_object(o)

    # Join the remaining big objects together.
    C.view_layer.objects.active = C.selected_objects[0]
    if len(C.selected_objects) > 1:
        bpy.ops.object.join()

    # Remove the mesh data that is not used.
    for m in D.meshes:
        if m.users == 0:
            D.meshes.remove(m)

    return volume


def process_objects(obj1, obj2):
    """Combine 2 objects into 1 and remove the 2 objects aftwerwards."""
    ob1 = remove_small_islands(obj1)
    ob2 = remove_small_islands(obj2)

    # Make a new bmesh that combines data from obj1 and obj2.
    combi = bmesh.new()
    combi.from_mesh(ob1.data)
    upper_boundaries = [v for v in combi.verts if v.is_boundary]
    combi.from_mesh(ob2.data)

    combi = stitch_meshes(combi, upper_boundaries)

    # Write data to obj1.
    combi.to_mesh(ob1.data)
    combi.free()
    volume = fill_holes_and_calc_volume(ob1)

    # Remove obj2 and update view layer.
    D.meshes.remove(ob2.data)
    C.view_layer.update()

    return volume * 1000000


def update_indices(upper_mesh, lower_mesh, prefix_name):
    """Update the upper- and lower mesh indices, based on the amount of
    triangle mesh volume objects are created."""
    volumes_obj_count = 0
    index = 0

    # Go over all objects.
    for obj in D.objects:
        # If object is a generated volume, increment volume objects counter.
        # Else increment index. If index is the same as upper or lower, set
        # correct index and return them both.
        if obj.name.startswith(prefix_name):
            volumes_obj_count += 1
        else:
            if index == upper_mesh:
                new_up = index + volumes_obj_count

            if index == lower_mesh:
                new_low = index + volumes_obj_count

            index += 1

    return new_up, new_low


def create_object_set_active(mesh_index, prefix_name):
    """Create a new object and set it active in edit mode."""
    # Make a new object which copies data from the object at index mesh_index.
    obj = D.objects[mesh_index].copy()
    obj.data = D.objects[mesh_index].data.copy()
    obj.data.name = prefix_name
    obj.name = prefix_name

    # Give the new_object a new material color.
    new_color = D.materials.new("Layer_color")
    new_color.diffuse_color = RGBAS[mesh_index % len(RGBAS)]
    obj.data.materials[0] = new_color

    # Link the object to the collection and update the view_layer.
    C.collection.objects.link(obj)
    C.view_layer.objects.active = obj
    C.view_layer.update()
    bpy.ops.object.mode_set(mode="EDIT")
    return obj


def delete_verts_and_update(bm, me, del_verts):
    """Delete the vertices specified in the list from the bmesh bm and update
    the mesh me."""
    bmesh.ops.delete(bm, geom=del_verts, context="VERTS")
    bmesh.update_edit_mesh(me)
    bpy.ops.object.mode_set(mode="OBJECT")


def cut_mesh(minx, maxx, miny, maxy, upper_mesh, lower_mesh, threshold,
             max_distance, dr="down"):
    """Remove vertices that are too close to the other mesh."""
    prefix_name = "triangle_mesh_volume"
    del_verts = []
    upper_mesh, lower_mesh = update_indices(upper_mesh, lower_mesh,
                                            prefix_name)

    # Link a copy of upper mesh to the scene, make it active and set it to
    # edit mode.
    obj = create_object_set_active(upper_mesh, prefix_name)

    # Make a bmesh object from the actice mesh.
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # For every vertex in the mesh, add it to the del_vertex list if it is
    # close to other mesh.
    for v in bm.verts:
        if (v.co[0] < minx or v.co[0] > maxx or v.co[1] < miny or
                v.co[1] > maxy):
            del_verts.append(v)
            continue

        if dr == "down":
            ray_shot = D.objects[lower_mesh].ray_cast(v.co, (0, 0, -1),
                                                      distance=max_distance)
        else:
            ray_shot = D.objects[lower_mesh].ray_cast(v.co, (0, 0, 1),
                                                      distance=max_distance)

        if (not ray_shot[0] or
                not dist_btwn_pts(ray_shot[1], v.co) >= threshold):
            del_verts.append(v)

    # Delete all vertices in the del_verts list and update the mesh.
    delete_verts_and_update(bm, me, del_verts)
    bm.free()
    return obj


def object_oriented_algorithm(
        meshes, threshold, minx, maxx, miny, maxy, minz, maxz, test=False):
    """Object-oriented algorithm to make a 3D model out of multiple triangular
    meshes."""
    tot_volume = 0
    max_distance = maxz - minz
    threshold /= 100

    # For all ordered mesh pairs from top to bottom, run the space-oriented
    # algorithm to get volume and list of primitives.
    for i, (upper_mesh, lower_mesh) in enumerate(zip(meshes[:-1], meshes[1:])):
        upper = cut_mesh(minx, maxx, miny, maxy, upper_mesh, lower_mesh,
                         threshold, max_distance, dr="down")
        lower = cut_mesh(minx, maxx, miny, maxy, lower_mesh, upper_mesh,
                         threshold, max_distance, dr="up")

        cur_volume = process_objects(upper, lower)
        print_layer_volume(False, cur_volume, i)
        tot_volume += cur_volume

    return tot_volume


# The main part of the code. Change variables and algorithms here.
def main():
    """Execute the script."""
    time_start = time.time()

    selected_meshes = C.selected_objects
    init()

    # Set info about threshold (in cm).
    threshold = 5

    # Determine bounding box and translate scene to origin.
    minx, maxx, miny, maxy, minz, maxz = bounding_box()
    difx, dify, difz = translate_to_origin(minx, maxx, miny, maxy, minz,
                                           maxz)

    # Update the new minimum and maximum coordinates.
    minx, maxx, miny, maxy, minz, maxz = update_minmax(
        minx, maxx, miny, maxy, minz, maxz, difx, dify, difz)

    # Method can be set to 'space' or 'object'.
    method = "space"

    if method == "space":
        # Specify the primitive type to be used to be 'tetra' or 'cuboid'.
        primitive = "cuboid"

        # Size in centimeters for the primitives.
        length, width, height = 5, 5, 2.5
        print_primitive_size(length, width, height)

        # Option to turn off drawing, it gives a minor performance boost.
        draw = True

        # Decide if you want a pointcloud exported so you are able to see
        # a solid object.
        export_ply_file = False

        # Specify place to save PLY-file and its name. If you want the PLY-file
        # to be saved in the home directory, set path to "". Otherwise specify
        # directories and end with a /.
        path_to_file = "Downloads/"
        ply_file_name = "testing_draw_23052020.ply"

        # Run the algorithm.
        volume = space_oriented_algorithm(
            ordered_meshes(selected_meshes), length, width, height, threshold,
            minx, maxx, miny, maxy, minz, maxz, primitive=primitive,
            export=export_ply_file, file_path=path_to_file,
            ply_name=ply_file_name, draw=draw)
    elif method == "object":
        # Run the object-oriented algorithm.
        volume = object_oriented_algorithm(
            ordered_meshes(selected_meshes), threshold, minx, maxx, miny, maxy,
            minz, maxz)

    print_total_volume(volume)

    print("Script Finished: %.4f sec.\n" % (time.time() - time_start))


if __name__ == "__main__":
    main()

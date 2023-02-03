#!/usr/bin/env python3
import sys
import numpy as np
import open3d as open3d

def point_cloud(file):
    print("Load the XYZ file of the point cloud, print it, and render it")
    point_cloud = open3d.io.read_point_cloud(file)
    R = point_cloud.get_rotation_matrix_from_xyz((0.7 * np.pi, 0, 0.6 * np.pi))
    point_cloud=point_cloud.rotate(R, center=(0,0,0))
    open3d.visualization.draw_geometries([point_cloud])
    return point_cloud

def down_sample(point_cloud):
    print("Downsample the point cloud with a voxel of 5.0")
    down_sample = point_cloud.voxel_down_sample(voxel_size=10.0)
    open3d.visualization.draw_geometries([down_sample])
    return down_sample


def surface_normal(down_sample,radius,neighbors):
    print("Recompute the normal of the downsampled point cloud")
    down_sample.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=int(radius), max_nn=int(neighbors)))
    down_sample.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    open3d.visualization.draw_geometries([down_sample],point_show_normal=True)
    return down_sample

def triangulation(normals,radius,triangulation_down,ball_radius):
    down_sample = normals.voxel_down_sample(voxel_size=int(triangulation_down))
    print("Triangle mesh with ball pivoting")
    rec_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(down_sample, open3d.utility.DoubleVector([int(ball_radius)])) 
    open3d.visualization.draw_geometries([rec_mesh], mesh_show_back_face=True)
    


if __name__ == "__main__":
    if len(sys.argv) == 7:
        radius = str(sys.argv[1])
        neighbors = str(sys.argv[2])
        triangulation_down=str(sys.argv[3])
        ball_radius=str(sys.argv[4])
        file=str(sys.argv[5])
        output_name=str(sys.argv[6])
        point_cloud = point_cloud(file)
        down_sample = down_sample(point_cloud)
        normals=surface_normal(down_sample,radius,neighbors)
        open3d.io.write_point_cloud(output_name, normals)
        triangulation(normals,radius,triangulation_down,ball_radius)
    else:
        print("There is a missing parameter")

    
        
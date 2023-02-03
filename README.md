This assignment consists of 3 main codes to evaluate the naive approach and the dynamic progamming approach for displaying the disparity map as well as the third one is for computing the surface normal and 3D triangulation.

* In order to run the naive program, the terminal should be written in it:
```
./OpenCV_naive_stereo (image_1 path) (image_2 path) (name of the output file) window_size
```

* Although to run the dynamic programming program, the terminal should be written in it:
```
./OpenCV_dp_stereo (image_1 path) (image_2 path) (name of the output file) window_size lambada
```

* To run the code of computing the normals and 3D triangulation:
```
./surface_normal.py (radius of KDTreeSearch to compute normals) (number of neighboors) 
(downsampling voxel size) (radius of ball pivoting) (path of the point cloud generated from DP) 
(name of the output file with extension xyzn)
```

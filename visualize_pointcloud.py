import open3d as o3d


## To read the pointcloud, just pass the file here:
pcd = o3d.io.read_point_cloud("point_clouds\sicilia\point_cloud.ply")
print(pcd)
o3d.visualization.draw_geometries([pcd])
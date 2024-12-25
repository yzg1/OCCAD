import os
import numpy as np
import trimesh

def ray_mesh_intersections(mesh, point):
    """
    Count the number of triangle intersections for a point using a z-axis ray
    to determine if the point is inside or outside the mesh
    """
    ray_origins = np.array([point])
    ray_directions = np.array([[0, 0, 1]])
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins, ray_directions
    )
    return len(index_tri) % 2 == 1  # Odd number of intersections means inside

def process_mesh(mesh_path, num_samples=100000, padding=0.05):
    """
    Process a mesh file to generate point samples and compressed occupancy information
    """
    # Load mesh
    mesh = trimesh.load_mesh(mesh_path)
    
    # Center and rescale mesh
    mesh.apply_translation(-mesh.bounding_box.centroid)
    longest_edge = np.max(mesh.bounding_box.extents)
    mesh.apply_scale(1.0 / longest_edge)
    
    # Generate uniform samples in unit cube with padding
    points = np.random.uniform(
        low=-0.5-padding, 
        high=0.5+padding, 
        size=(num_samples, 3)
    )
    
    # Determine occupancy for the entire point set
    occupancies = np.array([ray_mesh_intersections(mesh, point) for point in points])
    
    # Subsample occupancies and not points
    # indices = np.random.choice(num_samples, subsampled_size, replace=False)
    # occupancies_subsampled = occupancies[indices]
    
    # Compress occupancies
    compressed_occupancies = np.packbits(occupancies.astype(np.uint8))
    
    # Surface sampling
    surface_points, _ = trimesh.sample.sample_surface_even(mesh, num_samples)
    
    return {
        'points': points,
        'occupancies': compressed_occupancies,
        'loc': mesh.bounding_box.centroid,
        'scale': longest_edge
    }, surface_points

def main(input_dir, output_dir):
    """
    Process all OBJ files in the input directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter for processed and skipped files
    processed_count = 0
    skipped_count = 0
    
    # Walk through input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.obj'):
                # Construct full file paths
                mesh_path = os.path.join(root, file)
                
                # Create corresponding output subdirectory
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Generate filename
                filename_base = os.path.splitext(file)[0]
                points_path = os.path.join(output_subdir, f'points.npz')
                pointcloud_path = os.path.join(output_subdir, f'pointcloud.npz')
                
                # Check if both output files already exist
                if os.path.exists(points_path) and os.path.exists(pointcloud_path):
                    print(f"Skipping {mesh_path} (files already exist)")
                    skipped_count += 1
                    continue
                
                try:
                    # Process mesh
                    points_data, surface_points = process_mesh(mesh_path)
                    
                    # Save points.npz
                    np.savez(points_path, **points_data)
                    
                    # Save pointcloud.npz
                    np.savez(pointcloud_path, points=surface_points)
                    
                    print(f"Processed {mesh_path}")
                    processed_count += 1
                
                except Exception as e:
                    print(f"Error processing {mesh_path}: {e}")
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"Processed files: {processed_count}")
    print(f"Skipped files: {skipped_count}")

if __name__ == '__main__':
    input_dir = r'F:/obj/abc_watertight'
    output_dir = r'C:/Users/25581/Desktop/points/pc_occ'
    main(input_dir, output_dir)

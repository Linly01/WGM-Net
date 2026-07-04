import numpy as np
import pandas as pd
import networkx as nx
import os


def generate_geolocation_from_distance(distance_file, num_nodes=170):
    df = pd.read_csv(distance_file)
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    for _, row in df.iterrows():
        from_node = int(row['from'])
        to_node = int(row['to'])
        cost = float(row['cost'])
        G.add_edge(from_node, to_node, weight=cost)
    print("Generating node positions using force-directed layout...")
    pos = nx.spring_layout(G, weight='weight', iterations=100, seed=42)
    geolocation_data = np.array([pos[i] for i in range(num_nodes)])
    # 假设起始点为洛杉矶附近的坐标
    base_lon = -117.2898  # 经度
    base_lat = 34.1083  # 纬度
    scale_factor = 0.01
    geolocation_data[:, 0] = base_lon + geolocation_data[:, 0] * scale_factor * 20
    geolocation_data[:, 1] = base_lat + geolocation_data[:, 1] * scale_factor * 20

    return geolocation_data


def add_geolocation_to_data(original_data, geolocation_data):
    num_time_steps, num_nodes, num_features = original_data.shape
    extended_data = np.zeros((num_time_steps, num_nodes, num_features + 2))
    extended_data[:, :, :num_features] = original_data
    print("Adding geolocation data to each time step...")
    for t in range(num_time_steps):
        extended_data[t, :, num_features:] = geolocation_data

    return extended_data


def main():
    distance_file = r"D:\PostGraduate\deep learning code\PM-DMNet-main0\data\PEMS08\PEMS08.csv"
    data_file = r"D:\PostGraduate\deep learning code\PM-DMNet-main0\data\PEMS08\PEMS08.npz"
    output_dir = r"D:\PostGraduate\deep learning code\PM-DMNet-main0\data\PEMS08"
    if not os.path.exists(distance_file):
        print(f"Error: Distance file not found at {distance_file}")
        return

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return
    print("Step 1: Generating geolocation data from distance information...")
    geolocation_data = generate_geolocation_from_distance(distance_file)
    print("Step 2: Loading original data...")
    data = np.load(data_file)
    original_data = data['data']
    print(f"Original data shape: {original_data.shape}")
    print("Step 3: Adding geolocation dimension to data...")
    extended_data = add_geolocation_to_data(original_data, geolocation_data)
    print(f"Extended data shape: {extended_data.shape}")
    print(f"New feature dimensions: [flow, occupancy, speed, longitude, latitude]")
    print("Step 4: Saving extended data...")
    output_file = os.path.join(output_dir, "PEMS08_with_geolocation.npz")
    np.savez(output_file, data=extended_data)
    geolocation_df = pd.DataFrame(geolocation_data, columns=['longitude', 'latitude'])
    geolocation_df['node_id'] = range(len(geolocation_df))
    geolocation_df = geolocation_df[['node_id', 'longitude', 'latitude']]
    geolocation_csv = os.path.join(output_dir, "PEMS08_geolocation.csv")
    geolocation_df.to_csv(geolocation_csv, index=False)

    print("Process completed!")
    print(f"Geolocation data saved to: {geolocation_csv}")
    print(f"Extended dataset saved to: {output_file}")

    # 显示前几个节点的经纬度示例
    print("\nSample geolocation data (first 10 nodes):")
    print(geolocation_df.head(10))


if __name__ == "__main__":
    main()
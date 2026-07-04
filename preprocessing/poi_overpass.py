import numpy as np
import pandas as pd
import osmnx as ox
import shutil
import os
import time
import warnings

from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


# =====================
# 清除缓存并重新开始
# =====================
print("正在清除 OSMnx 缓存...")
cache_dir = ox.settings.cache_folder
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"已清除缓存目录: {cache_dir}")

# 禁用缓存以确保重新下载
ox.settings.use_cache = False
# 增加超时时间
ox.settings.timeout = 30
# 启用详细日志
ox.settings.log_console = False


# =====================
# 1️⃣ 加载 PEMS04 节点经纬度
# =====================
npz_file = "D:\PostGraduate\deep learning code\PM-DMNet-main0\data\PEMS08\PEMS08_with_geolocation.npz"
data = np.load(npz_file, allow_pickle=True)

print("文件里的数组:", data.files)
arr_name = data.files[0]
arr = data[arr_name]
print("数组形状:", arr.shape)

# 提取节点经纬度
node_lon = arr[0, :, 3]
node_lat = arr[0, :, 4]

# 验证前几个节点
print("\n验证前5个节点:")
for i in range(5):
    print(f"节点 {i}: 经度={node_lon[i]:.6f}, 纬度={node_lat[i]:.6f}")

# =====================
# 2️⃣ 功能分类定义
# =====================
radius = 500

poi_function_map = {
    "education": ["school", "university", "kindergarten"],
    "medical": ["hospital", "clinic", "pharmacy"],
    "residential": ["residential", "apartments"],
    "transport": ["bus_station", "tram_stop", "parking", "fuel", "taxi", "traffic_signals"],
    "commercial": ["restaurant", "cafe", "supermarket", "convenience", "bank", "atm", "hotel"],
    "leisure": ["park", "sports_centre", "theatre"]
}


poi_function_categories = list(poi_function_map.keys())
poi_features = []
global_poi_counts = {func: 0 for func in poi_function_categories}
# 指定的 23 类 POI
#poi_label_map = {
#    "school": [("amenity", "school")],
#    "university": [("amenity", "university")],
#    "kindergarten": [("amenity", "kindergarten")],
#    "hospital": [("amenity", "hospital")],
#    "clinic": [("amenity", "clinic")],
#    "pharmacy": [("amenity", "pharmacy")],

#    "residential": [("building", "residential")],
#    "apartments": [("building", "apartments")],

#    "bus_station": [("amenity", "bus_station")],
#    "tram_stop": [("railway", "tram_stop")],
#   "parking": [("amenity", "parking")],
#   "fuel": [("amenity", "fuel")],
#   "taxi": [("amenity", "taxi")],
#    "traffic_signals": [("highway", "traffic_signals")],

#    "restaurant": [("amenity", "restaurant")],
#    "cafe": [("amenity", "cafe")],
#    "supermarket": [("shop", "supermarket")],
#    "convenience": [("shop", "convenience")],
#    "bank": [("amenity", "bank")],
#    "atm": [("amenity", "atm")],
#    "hotel": [("tourism", "hotel")],

#    "park": [("leisure", "park")],
#   "sports_centre": [("leisure", "sports_centre")],
#    "theatre": [("amenity", "theatre"), ("leisure", "theatre")]
#}

#poi_categories = list(poi_label_map.keys())
#poi_features = []
#global_poi_counts = {k: 0 for k in poi_categories}

# =====================
# 3️⃣ 重新开始抓取每个节点 POI
# =====================
print(f"\n开始抓取 {len(node_lon)} 个节点的POI数据...")

for i, (lon_i, lat_i) in enumerate(zip(node_lon, node_lat)):
    print(f"\n=== 处理节点 {i} (索引 {i}) ===")
    print(f"经纬度: ( {lon_i:.6f},{lat_i:.6f})")

    tags = {"amenity": True, "shop": True, "leisure": True, "building": True}

    # 添加重试机制
    max_retries = 3
    retry_delay = 5  # 秒

    for attempt in range(max_retries):
        try:
            poi_gdf = ox.geometries_from_point((lat_i, lon_i), tags=tags, dist=radius)
            print(f"成功获取节点 {i} 的POI数据，找到 {len(poi_gdf)} 个几何对象")
            break  # 如果成功，跳出重试循环
        except Exception as e:
            print(f"节点 {i} 抓取失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                # 每次重试增加等待时间
                retry_delay += 5
            else:
                print(f"节点 {i} 的所有 {max_retries} 次尝试都失败了")
                poi_gdf = pd.DataFrame()


    # 初始化此节点的 23 类 POI 统计
    func_count = {label: 0 for label in poi_function_categories}
    if not poi_gdf.empty:
        for label, value_list in poi_function_map.items():
            count = 0
            for osm_key in ["amenity", "shop", "leisure", "building"]:
                if osm_key in poi_gdf.columns:
                    values = poi_gdf[osm_key].value_counts()
                    for osm_value in value_list:
                        count += values.get(osm_value, 0)
            func_count[label] = count

    poi_features.append(func_count)
    for func in poi_function_categories:
        global_poi_counts[func] += func_count[func]

    total_poi = sum(func_count.values())
    print(f"节点 {i + 1}/{len(node_lon)}: 抓取到 {total_poi} 个 POI")

    # 打印每个节点的详细分类信息（移除了仅前5个节点的限制）
    func_counts_str = ", ".join([f"{func}={func_count[func]}" for func in poi_function_categories])
    print(f"节点 {i} 详细分类: {func_counts_str}")

    # 添加短暂延迟，避免对服务器造成过大压力
    time.sleep(1)

# =====================
# 4️⃣ 构建功能特征矩阵
# =====================
poi_matrix = np.array([[func_count[func] for func in poi_function_categories] for func_count in poi_features])

# =====================
# 5️⃣ 保存文件
# =====================
np.save("D:/PostGraduate/deep learning code/PM-DMNet-main0/data/PeMS08/PEMS08_poi_6.npy", poi_matrix)
print("\nPEMS08_poi_6.npy 已保存！")

# =====================
# 6️⃣ 查看信息
# =====================
print(f"\nPOI 功能特征矩阵形状: {poi_matrix.shape}  # (节点数, 功能类别数)")
print("功能类别名称:", poi_function_categories)
print("前5个节点功能 POI 特征示例:\n", poi_matrix[:5])
# 新增：打印所有节点的POI类型总数量
print("\n所有节点的POI类型总数量统计：")
for func in poi_function_categories:
    # 打印功能类别总数量及包含的具体标签
    labels = ", ".join(poi_function_map[func])
    print(f"{func} 总数量: {global_poi_counts[func]} (包含: {labels})")
print(f"所有类型POI的总数量: {sum(global_poi_counts.values())}")
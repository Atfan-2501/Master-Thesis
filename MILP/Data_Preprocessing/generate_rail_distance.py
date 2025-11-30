import osmnx as ox
import networkx as nx
import pandas as pd

# 1. Define your terminals with WGS84 coordinates (lat, lon)
terminals = {
    "aarau":       (47.39, 8.05),   
    "chiasso":     (45.83, 9.03),   
    "stabio":      (45.85, 8.94),   
    "visp":        (46.29, 7.88),   
    "basel":       (47.55, 7.59),   
    "zurich":      (47.37, 8.54),   
    "bern":        (46.94, 7.44),   
    "luzern":      (47.05, 8.31),   
    "lausanne": (46.50, 6.60),
    "geneva": (46.21, 6.14),
    "olten":       (47.35, 7.90),   
    "schaffhausen":(47.69, 8.63),   
    "interlaken":  (46.68, 7.86),   
    "fribourg":    (46.80, 7.16),   
    "winterthur":  (47.49, 8.72),   
}

print(f"Terminals: {list(terminals.keys())}")

# 2. Download the rail network for Switzerland from OSM
print("Downloading Swiss rail network from OpenStreetMap...")
try:
    # Use custom_filter for rail network instead of network_type
    G = ox.graph_from_place(
        "Switzerland", 
        custom_filter='["railway"~"rail|light_rail|subway"]',
        simplify=True
    )
    print(f"Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
except Exception as e:
    print(f"Error downloading graph: {e}")
    print("Trying alternative approach with bounding box...")
    # Fallback: use bounding box for Switzerland
    G = ox.graph_from_bbox(
        north=47.8, south=45.8, 
        east=10.5, west=5.9,
        custom_filter='["railway"~"rail|light_rail|subway"]',
        simplify=True
    )

# Convert to undirected graph for better connectivity
G = G.to_undirected()

# 3. Snap each terminal to the nearest rail node
print("Snapping terminals to nearest rail nodes...")
terminal_nodes = {}
for name, (lat, lon) in terminals.items():
    try:
        # OSMnx uses (x=lon, y=lat)
        node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        terminal_nodes[name] = node
    except Exception as e:
        print(f"Warning: Could not snap {name} to rail network: {e}")

print("Matched terminals to graph nodes:")
for name, node in terminal_nodes.items():
    print(f"  {name:15s} -> node {node}")

# 4. Compute shortest-path rail distances between all pairs
print("\nComputing shortest-path rail distances (meters)...")
names = list(terminal_nodes.keys())  # Only use successfully snapped terminals
dist_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

for i, origin in enumerate(names):
    for j, dest in enumerate(names):
        if origin == dest:
            dist_matrix.loc[origin, dest] = 0.0
            continue
        try:
            # Shortest path distance along rail network, weighted by 'length' (meters)
            d = nx.shortest_path_length(
                G,
                terminal_nodes[origin],
                terminal_nodes[dest],
                weight="length"
            )
            dist_matrix.loc[origin, dest] = d
        except nx.NetworkXNoPath:
            # No rail path found between these nodes
            print(f"  Warning: No path found between {origin} and {dest}")
            dist_matrix.loc[origin, dest] = None
        except Exception as e:
            print(f"  Error computing path {origin} -> {dest}: {e}")
            dist_matrix.loc[origin, dest] = None

print("\nRail distance matrix (meters):")
print(dist_matrix)

# Convert to kilometers for easier reading
dist_matrix_km = dist_matrix / 1000
print("\nRail distance matrix (kilometers):")
print(dist_matrix_km.round(2))

# 5. Save to Excel/CSV for use in your thesis pipeline
try:
    dist_matrix.to_excel("rail_distance_matrix_osm.xlsx")
    dist_matrix_km.to_excel("rail_distance_matrix_osm_km.xlsx")
    print("\nSaved distances to Excel files.")
except Exception as e:
    print(f"\nWarning: Could not save Excel files: {e}")
    print("Excel support requires openpyxl. Install with: pip install openpyxl")

dist_matrix.to_csv("rail_distance_matrix_osm.csv")
dist_matrix_km.to_csv("rail_distance_matrix_osm_km.csv")
print("Saved distances to CSV files.")
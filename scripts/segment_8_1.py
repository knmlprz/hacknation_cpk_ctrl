import laspy
import numpy as np
import open3d as o3d
import os
import gc

# --- KONFIGURACJA ---
np.random.seed(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "Chmura_zadanie.las")
TEMP_FILE = os.path.join(SCRIPT_DIR, "temp_remainder.las")

# Parametry globalne
GROUND_THRESHOLD = 0.30     
SAMPLE_SIZE = 100000        

# LIMIT BEZPIECZEŃSTWA RAM
# Jeśli klasa ma więcej punktów niż to, pomijamy szukanie sąsiadów (uznajemy obiekt za pewnik)
CLUSTER_POINT_LIMIT = 200000 

GROUND_MODEL = None 

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================

def save_las(points, colors, header_template, filename, classification_val):
    if len(points) == 0:
        print(f"   [INFO] Brak punktów dla klasy {classification_val}. Pomijam.")
        return

    out = laspy.LasData(header_template)
    out.points = points
    if colors is not None:
        out.red = colors[:, 0]
        out.green = colors[:, 1]
        out.blue = colors[:, 2]
        
    out.classification[:] = classification_val
    out.write(filename)
    print(f"   -> Zapisano: {os.path.basename(filename)} ({len(points)} pkt)")

def get_heights(xyz):
    [a, b, c, d] = GROUND_MODEL
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    if c < 0: dists = -dists
    return dists

# --- INTELIGENTNE FILTROWANIE Z BEZPIECZNIKIEM ---

def safe_cluster_filter(xyz, mask_candidates, eps, min_points):
    """
    Filtruje kandydatów metodą DBSCAN, ale TYLKO jeśli nie jest ich za dużo.
    Zapobiega błędom 'Killed' przy dużych obiektach.
    """
    count = np.sum(mask_candidates)
    
    # Jeśli nikogo nie ma, zwracamy pustą maskę
    if count == 0: 
        return mask_candidates
    
    # 1. BEZPIECZNIK: Jeśli obiekt jest ogromny (np. cały most), ufamy kolorowi i pomijamy sąsiadów
    if count > CLUSTER_POINT_LIMIT:
        print(f"      [SKIP CLUSTER] Zbyt duży obiekt ({count} pkt). Pomijam analizę sąsiadów (RAM Safe).")
        return mask_candidates

    # 2. Jeśli obiekt jest mały/średni -> Czyścimy go DBSCANem
    print(f"      [CLUSTERING] Precyzyjne czyszczenie szumu ({count} pkt)...")
    
    # Wyciągamy punkty (tylko kandydatów)
    indices = np.where(mask_candidates)[0]
    cand_xyz = xyz[indices]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cand_xyz)
    
    # Klastrowanie
    try:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except Exception as e:
        print(f"      [WARN] Błąd klastrowania: {e}. Zwracam surowe wyniki.")
        return mask_candidates
    
    # Punkty z etykietą >= 0 to obiekty, -1 to szum
    valid_local = (labels >= 0)
    
    # Aktualizujemy maskę (odrzucamy szum)
    final_mask = np.zeros(len(xyz), dtype=bool)
    final_mask[indices[valid_local]] = True
    
    removed = count - np.sum(final_mask)
    if removed > 0:
        print(f"      [CLEANED] Usunięto {removed} luźnych punktów (szum).")
        
    return final_mask

# --- ANALIZA KOLORU ---

def is_vegetation_color(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # Jasna zieleń, Ciemna zieleń, Brąz
    is_green = (g > r * 1.05) & (g > b * 0.95)
    is_dark_veg = (g >= r * 0.9) & (g > b) & (r < 18000)
    is_brown = (r > g) & (g > b) & (r < g * 2.0) & (r > 3000)
    return is_green | is_dark_veg | is_brown

def is_concrete_asphalt_color(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    std_dev = np.std(rgb, axis=1)
    is_grey = std_dev < 5000 
    is_bright = np.mean(rgb, axis=1) > 25000
    return is_grey | is_bright

def is_rail_color(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    std_dev = np.std(rgb, axis=1)
    return (std_dev < 4000) & (r < 20000) & (g < 20000) & (b < 20000)

def is_blue_water(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return ((b > r) & (b > g)) | ((r<6000)&(g<6000)&(b<6000))

# ==========================================
# LOGIKI FILTRUJĄCE (HYBRYDOWE)
# ==========================================

def logic_ground(xyz, rgb):
    global GROUND_MODEL
    # RANSAC na próbce
    sample = xyz if len(xyz) < SAMPLE_SIZE else xyz[np.random.choice(len(xyz), SAMPLE_SIZE, replace=False)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample)
    model, _ = pcd.segment_plane(distance_threshold=GROUND_THRESHOLD, ransac_n=3, num_iterations=1000)
    GROUND_MODEL = model
    [a, b, c, d] = model
    print(f"   [GEO] Model terenu: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    return np.abs(dists) < GROUND_THRESHOLD

def logic_noise(xyz, rgb):
    h = get_heights(xyz)
    return h < -2.0

def logic_bridge(xyz, rgb):
    """MOSTY: Wysoko + Szaro. Pomijamy klastrowanie jeśli most jest gigantem."""
    h = get_heights(xyz)
    mask = (h > 3.5) & is_concrete_asphalt_color(rgb) & (~is_vegetation_color(rgb))
    # Mosty są duże i ciągłe.
    return safe_cluster_filter(xyz, mask, eps=2.0, min_points=200)

def logic_high_veg(xyz, rgb):
    """DRZEWA: Wysoko + Roślinnie."""
    h = get_heights(xyz)
    mask = (h > 5.0) & is_vegetation_color(rgb)
    # Drzewa warto oczyścić z szumu (pojedynczych pikseli w powietrzu)
    return safe_cluster_filter(xyz, mask, eps=0.8, min_points=30)

def logic_rail(xyz, rgb):
    """TORY: Nisko + Ciemno."""
    h = get_heights(xyz)
    mask = (h > 0.1) & (h < 1.5) & is_rail_color(rgb) & (~is_vegetation_color(rgb))
    # Tory to wąskie linie, klastrowanie tutaj świetnie usuwa przypadkowe kamienie
    return safe_cluster_filter(xyz, mask, eps=0.5, min_points=20)

def logic_water(xyz, rgb):
    h = get_heights(xyz)
    mask = (h < 0.2) & is_blue_water(rgb)
    return safe_cluster_filter(xyz, mask, eps=0.5, min_points=100)

def logic_buildings(xyz, rgb):
    """BUDYNKI: Reszta wysokich."""
    h = get_heights(xyz)
    mask = (h > 2.5) 
    # Budynki to duże bryły.
    return safe_cluster_filter(xyz, mask, eps=1.0, min_points=100)

def logic_med_veg(xyz, rgb):
    h = get_heights(xyz)
    mask = (h >= 1.5) & (h <= 5.0) & is_vegetation_color(rgb)
    return safe_cluster_filter(xyz, mask, eps=0.5, min_points=20)

def logic_low_veg(xyz, rgb):
    h = get_heights(xyz)
    mask = (h > 0.2) & (h < 1.5) & is_vegetation_color(rgb)
    return safe_cluster_filter(xyz, mask, eps=0.3, min_points=10)

# ==========================================
# SILNIK (Pipeline)
# ==========================================

def process_step(input_path, output_class_name, class_code, filter_logic_function):
    print(f"\n--- Przetwarzanie: {output_class_name} (Kod {class_code}) ---")
    if not os.path.exists(input_path): return None

    try: las = laspy.read(input_path)
    except: return None

    # Pobieramy dane
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    rgb = np.vstack((las.red, las.green, las.blue)).transpose()
    
    # 1. Logika + Bezpieczne Klastrowanie
    mask = filter_logic_function(xyz, rgb)

    class_points = las.points[mask]
    class_colors = rgb[mask]
    remainder_points = las.points[~mask]
    remainder_colors = rgb[~mask]
    
    print(f"   Zapis: {len(class_points)} pkt | Reszta: {len(remainder_points)} pkt")

    # Zapis
    class_file = os.path.join(SCRIPT_DIR, f"Wynik_{class_code}_{output_class_name}.las")
    save_las(class_points, class_colors, las.header, class_file, class_code)

    temp_out = os.path.join(SCRIPT_DIR, "temp_next_step.las")
    save_las(remainder_points, remainder_colors, las.header, temp_out, 1)

    del las, xyz, rgb, mask, class_points, remainder_points
    gc.collect()

    if os.path.exists(input_path) and input_path != INPUT_FILE:
        os.remove(input_path)
    os.rename(temp_out, TEMP_FILE)
    
    return TEMP_FILE

def main():
    print("=== START KLASYFIKACJI (HYBRYDA Z BEZPIECZNIKIEM) ===")
    
    current = INPUT_FILE
    
    # Kolejność przetwarzania
    current = process_step(current, "Grunt", 2, logic_ground)
    current = process_step(current, "Szum", 7, logic_noise)
    current = process_step(current, "Woda", 9, logic_water)
    current = process_step(current, "Roslinnosc_Wysoka", 5, logic_high_veg)
    current = process_step(current, "Roslinnosc_Srednia", 4, logic_med_veg)
    current = process_step(current, "Roslinnosc_Niska", 3, logic_low_veg)
    current = process_step(current, "Tory", 18, logic_rail)
    current = process_step(current, "Mosty", 17, logic_bridge) # Teraz bezpieczne!
    current = process_step(current, "Budynki", 6, logic_buildings)
    
    if os.path.exists(current):
        final_file = os.path.join(SCRIPT_DIR, "Wynik_1_Niesklasyfikowane.las")
        las = laspy.read(current)
        save_las(las.points, np.vstack((las.red, las.green, las.blue)).transpose(), las.header, final_file, 1)
        os.remove(current)

    print("\nGOTOWE! Algorytm działał selektywnie (klastrował tylko małe obiekty).")

if __name__ == "__main__":
    main()
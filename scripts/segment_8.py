import laspy
import numpy as np
import open3d as o3d
import os
import gc
import argparse
import sys

# --- KONFIGURACJA ---
np.random.seed(42)

# Parametry globalne
GROUND_THRESHOLD = 0.30     
SAMPLE_SIZE = 100000        
CLUSTER_POINT_LIMIT = 200000 
GROUND_MODEL = None 

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================

def save_las(points, colors, header_template, filename, classification_val):
    if len(points) == 0:
        # Jeśli to plik wynikowy (klasyfikacja), to nie zapisujemy pustego.
        # Ale UWAGA: funkcja process_step musi zapisać plik tymczasowy (resztę) nawet jak jest pusta (choć to mało prawdopodobne).
        if "temp_" not in filename:
            print(f"   [INFO] Brak punktów dla klasy {classification_val}. Nie zapisuję pliku.", flush=True)
            return

    # Tworzymy nowy plik LAS
    out = laspy.LasData(header_template)
    out.points = points
    
    # Przypisujemy kolory, jeśli istnieją
    if colors is not None:
        out.red = colors[:, 0]
        out.green = colors[:, 1]
        out.blue = colors[:, 2]
        
    out.classification[:] = classification_val
    out.write(filename)
    if "temp_" not in filename:
        print(f"   -> Zapisano: {os.path.basename(filename)} ({len(points)} pkt)", flush=True)

def get_heights(xyz):
    """Oblicza wysokość punktów względem modelu terenu (płaszczyzny)."""
    if GROUND_MODEL is None:
        return xyz[:, 2] # Jeśli brak modelu, zwróć Z absolutne (fallback)
        
    [a, b, c, d] = GROUND_MODEL
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    
    # Normalizacja kierunku wektora normalnego (żeby góra była "na plusie")
    if c < 0: dists = -dists
    return dists

# --- INTELIGENTNE FILTROWANIE (DBSCAN) ---

def safe_cluster_filter(xyz, mask_candidates, eps, min_points):
    """
    Filtruje szum za pomocą klastrowania, ale pomija proces dla bardzo dużych obiektów
    (np. całe mosty), aby uniknąć braku pamięci RAM.
    """
    count = np.sum(mask_candidates)
    if count == 0: return mask_candidates
    
    # Zabezpieczenie przed przepełnieniem pamięci przy gigantycznych obiektach
    if count > CLUSTER_POINT_LIMIT:
        print(f"      [SKIP CLUSTER] Obiekt zbyt duży ({count} pkt) - pomijam klastrowanie (RAM Safe).", flush=True)
        return mask_candidates

    print(f"      [CLUSTERING] Precyzyjne czyszczenie szumu ({count} pkt)...", flush=True)
    indices = np.where(mask_candidates)[0]
    cand_xyz = xyz[indices]
    
    # Użycie Open3D do szybkiego klastrowania
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cand_xyz)
    
    try:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except Exception as e:
        print(f"      [WARN] Błąd klastrowania: {e}", flush=True)
        return mask_candidates
    
    # -1 oznacza szum w DBSCAN
    valid_local = (labels >= 0)
    final_mask = np.zeros(len(xyz), dtype=bool)
    final_mask[indices[valid_local]] = True
    return final_mask

# --- ANALIZA KOLORU ---

def is_vegetation_color(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # Heurystyki dla różnych odcieni zieleni i brązu
    is_green = (g > r * 1.05) & (g > b * 0.95)
    is_dark_veg = (g >= r * 0.9) & (g > b) & (r < 18000)
    is_brown = (r > g) & (g > b) & (r < g * 2.0) & (r > 3000)
    return is_green | is_dark_veg | is_brown

def is_concrete_asphalt_color(rgb):
    std_dev = np.std(rgb, axis=1)
    is_grey = std_dev < 5000 
    is_bright = np.mean(rgb, axis=1) > 25000
    return is_grey | is_bright

def is_rail_color(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    std_dev = np.std(rgb, axis=1)
    # Ciemne, mało nasycone (szare/rdzawe)
    return (std_dev < 4000) & (r < 20000) & (g < 20000) & (b < 20000)

def is_blue_water(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return ((b > r) & (b > g)) | ((r<6000)&(g<6000)&(b<6000))

# ==========================================
# LOGIKI FILTRUJĄCE
# ==========================================

def logic_ground(xyz, rgb):
    global GROUND_MODEL
    # Próbkowanie dla szybkości RANSAC
    sample = xyz if len(xyz) < SAMPLE_SIZE else xyz[np.random.choice(len(xyz), SAMPLE_SIZE, replace=False)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample)
    
    # Wyznaczanie płaszczyzny terenu
    model, _ = pcd.segment_plane(distance_threshold=GROUND_THRESHOLD, ransac_n=3, num_iterations=1000)
    GROUND_MODEL = model
    [a, b, c, d] = model
    print(f"   [GEO] Wykryto teren: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0", flush=True)
    
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    return np.abs(dists) < GROUND_THRESHOLD

def logic_noise(xyz, rgb):
    h = get_heights(xyz)
    return h < -2.0  # Punkty głęboko pod ziemią

def logic_bridge(xyz, rgb):
    h = get_heights(xyz)
    # Wysoko + beton/asfalt + brak roślinności
    mask = (h > 3.5) & is_concrete_asphalt_color(rgb) & (~is_vegetation_color(rgb))
    return safe_cluster_filter(xyz, mask, eps=2.0, min_points=200)

def logic_high_veg(xyz, rgb):
    h = get_heights(xyz)
    mask = (h > 5.0) & is_vegetation_color(rgb)
    return safe_cluster_filter(xyz, mask, eps=0.8, min_points=30)

def logic_rail(xyz, rgb):
    h = get_heights(xyz)
    # Nisko nad ziemią, specyficzny kolor szyn/podkładów
    mask = (h > 0.1) & (h < 1.5) & is_rail_color(rgb) & (~is_vegetation_color(rgb))
    return safe_cluster_filter(xyz, mask, eps=0.5, min_points=20)

def logic_water(xyz, rgb):
    h = get_heights(xyz)
    mask = (h < 0.2) & is_blue_water(rgb)
    return safe_cluster_filter(xyz, mask, eps=0.5, min_points=100)

def logic_buildings(xyz, rgb):
    h = get_heights(xyz)
    # Wszystko co wysokie i nie zostało wykryte jako drzewa/mosty
    mask = (h > 2.5) 
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
# SILNIK PRZETWARZANIA
# ==========================================

def process_step(input_path, output_path, class_code, filter_logic_function, step_index):
    print(f"\n--- Przetwarzanie: {os.path.basename(output_path)} ---", flush=True)
    
    if not os.path.exists(input_path): 
        print(f"[BŁĄD] Nie znaleziono pliku wejściowego: {input_path}", flush=True)
        return None

    try: 
        las = laspy.read(input_path)
    except Exception as e:
        print(f"[BŁĄD] Nie można otworzyć pliku LAS: {e}", flush=True)
        return None

    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    rgb = np.vstack((las.red, las.green, las.blue)).transpose()
    
    # Uruchomienie logiki filtrowania
    mask = filter_logic_function(xyz, rgb)

    class_points = las.points[mask]
    class_colors = rgb[mask]
    remainder_points = las.points[~mask]
    remainder_colors = rgb[~mask]
    
    print(f"   Znaleziono: {len(class_points)} pkt | Pozostało: {len(remainder_points)} pkt", flush=True)

    # Zapisz znalezioną klasę
    save_las(class_points, class_colors, las.header, output_path, class_code)

    # Zapisz resztę do NOWEGO pliku tymczasowego (unikalna nazwa!)
    temp_dir = os.path.dirname(input_path)
    temp_out_name = f"temp_processing_step_{step_index}.las"
    temp_out = os.path.join(temp_dir, temp_out_name)
    
    save_las(remainder_points, remainder_colors, las.header, temp_out, 1)

    # Czyszczenie pamięci
    del las, xyz, rgb, mask, class_points, remainder_points
    gc.collect()

    return temp_out

def main():
    # Definicja argumentów, które przesyła interfejs (interfejs.py)
    parser = argparse.ArgumentParser(description="Segmentacja chmur LAS - Worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_ground", required=True)
    parser.add_argument("--out_noise", required=True)
    parser.add_argument("--out_water", required=True)
    parser.add_argument("--out_rails", required=True)
    parser.add_argument("--out_bridges", required=True)
    parser.add_argument("--out_veg_high", required=True)
    parser.add_argument("--out_buildings", required=True)
    parser.add_argument("--out_veg_med", required=True)
    parser.add_argument("--out_veg_low", required=True)
    parser.add_argument("--out_unclassified", required=True)
    
    args = parser.parse_args()

    print(f"=== START ANALIZY: {os.path.basename(args.input)} ===", flush=True)
    
    current_temp = args.input
    original_input = args.input # Zapamiętujemy oryginał, żeby go nie usunąć
    step_idx = 1
    
    # 1. Najpierw GRUNT (niezbędny do relatywnej wysokości)
    next_temp = process_step(current_temp, args.out_ground, 2, logic_ground, step_idx)
    
    if next_temp:
        current_temp = next_temp
        step_idx += 1
    else:
        print("[CRITICAL] Nie udało się wyznaczyć gruntu. Przerywam.", flush=True)
        return

    # 2. Lista kolejnych kroków
    steps = [
        (args.out_noise, 7, logic_noise),
        (args.out_water, 9, logic_water),
        (args.out_rails, 18, logic_rail),
        (args.out_bridges, 17, logic_bridge),
        (args.out_veg_high, 5, logic_high_veg),
        (args.out_buildings, 6, logic_buildings),
        (args.out_veg_med, 4, logic_med_veg),
        (args.out_veg_low, 3, logic_low_veg)
    ]

    # Pętla przetwarzania
    for out_path, code, logic in steps:
        if current_temp:
            prev_temp = current_temp
            # Wykonaj krok z nowym indeksem
            new_temp = process_step(current_temp, out_path, code, logic, step_idx)
            
            if new_temp:
                current_temp = new_temp
                step_idx += 1
                
                # Usuwamy poprzedni plik tymczasowy, ale TYLKO jeśli nie jest to oryginał
                if prev_temp != original_input and os.path.exists(prev_temp):
                    try: 
                        os.remove(prev_temp)
                        # print(f"   [CLEAN] Usunięto tymczasowy: {os.path.basename(prev_temp)}")
                    except: pass
            else:
                print(f"[WARN] Błąd w kroku {out_path}, kontynuuję na starym pliku.", flush=True)

    # 3. Ostatni krok: To co zostało, to klasa "Niesklasyfikowane"
    if current_temp and os.path.exists(current_temp):
        print(f"\n--- Finalizacja: Zapis reszty ---", flush=True)
        try:
            las = laspy.read(current_temp)
            # Zapisujemy pod nazwą zdefiniowaną w interfejsie
            save_las(las.points, np.vstack((las.red, las.green, las.blue)).transpose(), las.header, args.out_unclassified, 1)
            las = None
            gc.collect()
            
            # Sprzątamy ostatni temp
            if current_temp != original_input:
                os.remove(current_temp)
        except Exception as e:
            print(f"[BŁĄD] Przy zapisie reszty: {e}", flush=True)

    print("\n[KONIEC PROCESU] Wszystkie pliki wygenerowane.", flush=True)

if __name__ == "__main__":
    main()
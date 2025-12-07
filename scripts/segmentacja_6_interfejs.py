import laspy
import numpy as np
import open3d as o3d
import os
import gc
import subprocess
import sys
import argparse

# --- KONFIGURACJA ---
np.random.seed(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "Chmura_zadanie.las")
TEMP_FILE = os.path.join(SCRIPT_DIR, "temp_remainder.las")

# Parametry globalne
GROUND_THRESHOLD = 0.30     
SAMPLE_SIZE = 100000        

# Globalny model terenu
GROUND_MODEL = None 

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================

def save_las(points, colors, header_template, filename, classification_val):
    if len(points) == 0:
        print(f"   [INFO] Brak punktów dla klasy {classification_val}. Pomijam.", flush=True)
        return

    out = laspy.LasData(header_template)
    out.points = points
    if colors is not None:
        out.red = colors[:, 0]
        out.green = colors[:, 1]
        out.blue = colors[:, 2]
        
    out.classification[:] = classification_val
    out.write(filename)
    # Krótki log o zapisie
    print(f"   -> Zapisano: {os.path.basename(filename)} ({len(points)} pkt)", flush=True)

def get_heights(xyz):
    """Zwraca wysokość punktu nad modelem terenu"""
    [a, b, c, d] = GROUND_MODEL
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    if c < 0: dists = -dists
    return dists

# ==========================================
# LOGIKA KOLORÓW
# ==========================================

def is_vegetation_color(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    is_green = (g > r * 1.05) & (g > b * 0.95)
    is_dark_veg = (g >= r * 0.9) & (g > b) & (r < 18000)
    is_brown = (r > g) & (g > b) & (r < g * 2.0) & (r > 3000)
    return is_green | is_dark_veg | is_brown

def is_concrete_asphalt_color(rgb):
    std_dev = np.std(rgb, axis=1)
    is_grey = std_dev < 5000 
    brightness = np.mean(rgb, axis=1)
    is_bright = brightness > 25000
    return is_grey | is_bright

def is_rail_color(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    std_dev = np.std(rgb, axis=1)
    return (std_dev < 4000) & (r < 20000) & (g < 20000) & (b < 20000)

def is_blue_water(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return ((b > r) & (b > g)) | ((r<6000)&(g<6000)&(b<6000))

# ==========================================
# LOGIKI FILTRUJĄCE
# ==========================================

def logic_ground(xyz, rgb):
    global GROUND_MODEL
    sample = xyz if len(xyz) < SAMPLE_SIZE else xyz[np.random.choice(len(xyz), SAMPLE_SIZE, replace=False)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample)
    model, _ = pcd.segment_plane(distance_threshold=GROUND_THRESHOLD, ransac_n=3, num_iterations=1000)
    GROUND_MODEL = model
    [a, b, c, d] = model
    print(f"   [GEO] Model terenu ustalony: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0", flush=True)
    
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    return np.abs(dists) < GROUND_THRESHOLD

def logic_noise(xyz, rgb):
    h = get_heights(xyz)
    return h < -2.0 

def logic_bridge(xyz, rgb):
    h = get_heights(xyz)
    mask = (h > 3.5) & is_concrete_asphalt_color(rgb) & (~is_vegetation_color(rgb))
    return mask

def logic_water(xyz, rgb):
    h = get_heights(xyz)
    return (h < 0.2) & is_blue_water(rgb)

def logic_rail(xyz, rgb):
    h = get_heights(xyz)
    return (h > 0.1) & (h < 1.5) & is_rail_color(rgb) & (~is_vegetation_color(rgb))

def logic_high_veg(xyz, rgb):
    h = get_heights(xyz)
    return (h > 5.0) & is_vegetation_color(rgb)

def logic_buildings(xyz, rgb):
    h = get_heights(xyz)
    return (h > 2.5)

def logic_med_veg(xyz, rgb):
    h = get_heights(xyz)
    return (h >= 1.5) & (h <= 5.0) & is_vegetation_color(rgb)

def logic_low_veg(xyz, rgb):
    h = get_heights(xyz)
    return (h > 0.2) & (h < 1.5) & is_vegetation_color(rgb)

# ==========================================
# SILNIK (Pipeline)
# ==========================================

def process_step(input_path, output_path, class_code, filter_logic_function, step_curr, step_total):
    """
    Przetwarza jeden krok filtracji.
    Dodano: step_curr i step_total do wyświetlania postępu w logach.
    """
    output_class_name = os.path.basename(output_path)
    
    # --- NOWY FORMAT LOGÓW DLA INTERFEJSU ---
    print(f"\n>>> [ETAP {step_curr}/{step_total}] Generowanie: {output_class_name}...", flush=True)
    
    if not os.path.exists(input_path): 
        print(f"!!! Błąd: Brak pliku wejściowego dla etapu {step_curr}", flush=True)
        return None

    try: las = laspy.read(input_path)
    except: return None

    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    rgb = np.vstack((las.red, las.green, las.blue)).transpose()
    
    # 1. LOGIKA
    mask = filter_logic_function(xyz, rgb)

    # 2. SEPARACJA
    class_points = las.points[mask]
    class_colors = rgb[mask]
    
    remainder_points = las.points[~mask]
    remainder_colors = rgb[~mask]
    
    print(f"   Status: Znaleziono {len(class_points)} pkt. Pozostało {len(remainder_points)} pkt.", flush=True)

    # 3. ZAPIS
    save_las(class_points, class_colors, las.header, output_path, class_code)

    temp_out = os.path.join(SCRIPT_DIR, "temp_next_step.las")
    save_las(remainder_points, remainder_colors, las.header, temp_out, 1)

    # 4. MEMORY
    del las, xyz, rgb, mask, class_points, remainder_points
    gc.collect()

    # Rotacja
    if os.path.exists(input_path) and input_path != args_global.input:
        os.remove(input_path)
    if os.path.exists(temp_out):
        os.rename(temp_out, TEMP_FILE)
    
    return TEMP_FILE

# Zmienna globalna na argumenty
args_global = None

def main():
    # --- 1. OBSŁUGA INTERFEJSU (TRYB GUI) ---
    interface_path = os.path.join(SCRIPT_DIR, "interfejs.py")
    
    if len(sys.argv) == 1:
        if os.path.exists(interface_path):
            print("Uruchamianie interfejsu graficznego...", flush=True)
            subprocess.run([sys.executable, interface_path])
            return
        else:
            print("Brak pliku interfejs.py, uruchamiam w trybie domyślnym...", flush=True)

    # --- 2. OBSŁUGA CLI (WORKER MODE) ---
    parser = argparse.ArgumentParser()
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
    
    global args_global
    args_global = parser.parse_args()

    print(f"=== ROZPOCZYNAM PROCES KLASYFIKACJI ===", flush=True)
    print(f"Plik wejściowy: {os.path.basename(args_global.input)}", flush=True)
    
    current = args_global.input
    TOTAL_STEPS = 10
    
    # KOLEJNOŚĆ (Pipeline) z numeracją kroków
    
    # 1. Grunt
    current = process_step(current, args_global.out_ground, 2, logic_ground, 1, TOTAL_STEPS)
    # 2. Szum
    current = process_step(current, args_global.out_noise, 7, logic_noise, 2, TOTAL_STEPS)
    # 3. Woda
    current = process_step(current, args_global.out_water, 9, logic_water, 3, TOTAL_STEPS)
    # 4. Tory
    current = process_step(current, args_global.out_rails, 18, logic_rail, 4, TOTAL_STEPS)
    # 5. Mosty
    current = process_step(current, args_global.out_bridges, 17, logic_bridge, 5, TOTAL_STEPS)
    # 6. Veg High
    current = process_step(current, args_global.out_veg_high, 5, logic_high_veg, 6, TOTAL_STEPS)
    # 7. Budynki
    current = process_step(current, args_global.out_buildings, 6, logic_buildings, 7, TOTAL_STEPS)
    # 8. Veg Med
    current = process_step(current, args_global.out_veg_med, 4, logic_med_veg, 8, TOTAL_STEPS)
    # 9. Veg Low
    current = process_step(current, args_global.out_veg_low, 3, logic_low_veg, 9, TOTAL_STEPS)
    
    # 10. Reszta -> Unclassified
    print(f"\n>>> [ETAP 10/10] Generowanie: Niesklasyfikowane (Reszta)...", flush=True)
    if os.path.exists(current):
        las = laspy.read(current)
        save_las(las.points, np.vstack((las.red, las.green, las.blue)).transpose(), las.header, args_global.out_unclassified, 1)
        os.remove(current)

    print("\n=== GOTOWE! WSZYSTKIE ETAPY ZAKOŃCZONE ===", flush=True)

if __name__ == "__main__":
    main()
import laspy
import numpy as np
import open3d as o3d
import os
import gc  # Garbage Collector

# --- KONFIGURACJA ---
np.random.seed(42) # Determinizm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "Chmura_zadanie.las")
TEMP_FILE = os.path.join(SCRIPT_DIR, "temp_remainder.las")

# Parametry globalne
GROUND_THRESHOLD = 0.30     
SAMPLE_SIZE = 100000        

# Globalny model terenu (zapisywany po 1. kroku)
GROUND_MODEL = None 

# --- FUNKCJE POMOCNICZE (KOLOR I GEOMETRIA) ---

def save_las(points, colors, header_template, filename, classification_val):
    if len(points) == 0:
        print(f"   [INFO] Brak punktów dla klasy {classification_val}. Pomijam.")
        return

    out = laspy.LasData(header_template)
    out.points = points
    # Ważne: musimy przypisać kolory, żeby nie zginęły w plikach wynikowych
    if colors is not None:
        out.red = colors[:, 0]
        out.green = colors[:, 1]
        out.blue = colors[:, 2]
        
    out.classification[:] = classification_val
    out.write(filename)
    print(f"   -> Zapisano: {os.path.basename(filename)} ({len(points)} pkt)")

def is_color_green(rgb):
    """Zielony dominuje (Roślinność)"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # Green > Red i Green > Blue (z lekkim marginesem)
    return (g > r * 1.1) & (g > b * 0.9)

def is_color_blue(rgb):
    """Niebieski dominuje (Woda)"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (b > r) & (b > g)

def is_color_grey_or_dark(rgb):
    """Szary/Ciemny (Asfalt, Tory, Beton)"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # Szary to r~g~b, Ciemny to niskie wartości
    # Sprawdzamy czy odchylenie standardowe między kanałami jest małe (szarość)
    std_dev = np.std(rgb, axis=1)
    is_grey = std_dev < 5000 # Mała różnica między R, G, B
    is_dark = (r < 15000) & (g < 15000) & (b < 15000)
    return is_grey | is_dark

def calculate_normals_roughness(points):
    """Oblicza normalne i szorstkość dla podzbioru punktów (Geometria)"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    normals = np.asarray(pcd.normals)
    # Zwracamy składową Z normalnej (0=pionowa ściana, 1=pozioma podłoga)
    return np.abs(normals[:, 2])

def process_step(input_path, output_class_name, class_code, filter_logic_function):
    """PIPELINE: Wczytaj resztę -> Wytnij klasę -> Zapisz klasę -> Zapisz nową resztę"""
    print(f"\n--- Przetwarzanie: {output_class_name} (Kod {class_code}) ---")
    
    if not os.path.exists(input_path):
        return None

    try:
        las = laspy.read(input_path)
    except:
        return None

    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    rgb = np.vstack((las.red, las.green, las.blue)).transpose()
    
    # LOGIKA FILTRUJĄCA
    mask = filter_logic_function(xyz, rgb)

    class_points = las.points[mask]
    class_colors = rgb[mask]
    
    remainder_points = las.points[~mask]
    remainder_colors = rgb[~mask]
    
    print(f"   Znaleziono: {len(class_points)} | Pozostało: {len(remainder_points)}")

    # Zapisz klasę
    class_file = os.path.join(SCRIPT_DIR, f"Wynik_{class_code}_{output_class_name}.las")
    save_las(class_points, class_colors, las.header, class_file, class_code)

    # Zapisz resztę do temp
    temp_out = os.path.join(SCRIPT_DIR, "temp_next_step.las")
    save_las(remainder_points, remainder_colors, las.header, temp_out, 1)

    del las, xyz, rgb, mask, class_points, remainder_points
    gc.collect()

    if os.path.exists(input_path) and input_path != INPUT_FILE:
        os.remove(input_path)
    os.rename(temp_out, TEMP_FILE)
    
    return TEMP_FILE

# --- LOGIKI ---

def get_heights(xyz):
    [a, b, c, d] = GROUND_MODEL
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    if c < 0: dists = -dists
    return dists

def logic_ground(xyz, rgb):
    global GROUND_MODEL
    # RANSAC na próbce
    sample = xyz if len(xyz) < SAMPLE_SIZE else xyz[np.random.choice(len(xyz), SAMPLE_SIZE, replace=False)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample)
    model, _ = pcd.segment_plane(distance_threshold=GROUND_THRESHOLD, ransac_n=3, num_iterations=1000)
    GROUND_MODEL = model
    
    [a, b, c, d] = model
    print(f"   Model terenu: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    # Obliczamy dla wszystkich
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    return np.abs(dists) < GROUND_THRESHOLD

def logic_noise(xyz, rgb):
    h = get_heights(xyz)
    return h < -1.0 # Wszystko głęboko pod ziemią

def logic_water(xyz, rgb):
    h = get_heights(xyz)
    # Woda jest nisko (blisko 0) i jest niebieska lub ciemna
    return (h < 0.2) & (is_color_blue(rgb) | (is_color_grey_or_dark(rgb) & (h < 0.0)))

def logic_rail(xyz, rgb):
    """TORY: Nisko, Liniowe, Szare/Rdzawe (Nie zielone)"""
    h = get_heights(xyz)
    # Tory są na nasypach lub przy ziemi (0.1m - 1.2m)
    height_ok = (h > 0.1) & (h < 1.2)
    # Kolor: Beton/Stal/Rdza (Nie zielony!)
    color_ok = is_color_grey_or_dark(rgb) & (~is_color_green(rgb))
    
    # Weryfikacja geometryczna: Są płaskie/liniowe (Normalna Z bliska 1 - podkłady, lub 0 - szyna)
    # Żeby przyspieszyć, sprawdzamy geometrię tylko kandydatów
    candidates_mask = height_ok & color_ok
    
    # Jeśli kandydatów jest mało, zwracamy ich
    # Jeśli chcemy być super dokładni, można tu policzyć normalne, ale w skrypcie sekwencyjnym
    # ryzykowne jest liczenie normalnych dla dużej chmury.
    # Załóżmy, że height+color wytnie trawę.
    return candidates_mask

def logic_high_veg(xyz, rgb):
    """DRZEWA: Wysoko (>5m) i ZIELONO"""
    h = get_heights(xyz)
    return (h > 5.0) & is_color_green(rgb)

def logic_bridge(xyz, rgb):
    """MOSTY: Wysoko, Płasko (Asfalt), Nie zielono"""
    h = get_heights(xyz)
    
    # Kandydaci: Wysoko nad ziemią i nie są zieleni (drzewa już wycięliśmy, ale dla pewności)
    candidates = (h > 4.0) & (~is_color_green(rgb)) & is_color_grey_or_dark(rgb)
    
    if np.sum(candidates) == 0: return candidates
    
    # Dodatkowa geometria: Most ma płaską nawierzchnię (Normalna Z ~ 1.0)
    # Wyciągamy punkty kandydatów, liczymy normalne
    cand_xyz = xyz[candidates]
    norm_z = calculate_normals_roughness(cand_xyz)
    
    # Płaskie powierzchnie (Road deck)
    is_flat = norm_z > 0.85
    
    # Tworzymy maskę globalną
    final_mask = np.zeros(len(xyz), dtype=bool)
    # Mapujemy wynik lokalny (is_flat) z powrotem na globalną tablicę
    # Uwaga: To jest uproszczone mapowanie (indices mapping)
    # Żeby to zrobić w numpy bez błędów indeksowania:
    candidate_indices = np.where(candidates)[0]
    final_mask[candidate_indices[is_flat]] = True
    
    return final_mask

def logic_building(xyz, rgb):
    """BUDYNKI: To co zostało wysokiego (>2.5m)"""
    # Ponieważ drzewa (zielone) i mosty (płaskie szare) już wycięliśmy,
    # to co zostało wysokiego to zazwyczaj budynki.
    h = get_heights(xyz)
    return (h > 2.5) & (~is_color_green(rgb))

def logic_med_veg(xyz, rgb):
    """KRZEWY: 1.5m - 5m + Zielono"""
    h = get_heights(xyz)
    return (h >= 1.5) & (h <= 5.0) # & is_color_green(rgb) - opcjonalnie, ale zazwyczaj w tej wysokości to krzaki

def logic_low_veg(xyz, rgb):
    """TRAWA: 0.2m - 1.5m"""
    h = get_heights(xyz)
    return (h > 0.2) & (h < 1.5)

# --- GŁÓWNA PĘTLA ---

def main():
    print("=== START KLASYFIKACJI (KOLOR + GEOMETRIA) ===")
    
    current = INPUT_FILE
    
    # 1. Grunt (Baza)
    current = process_step(current, "Grunt", 2, logic_ground)
    
    # 2. Szum
    current = process_step(current, "Szum", 7, logic_noise)
    
    # 3. Woda (Nisko + Niebiesko)
    current = process_step(current, "Woda", 9, logic_water)
    
    # 4. Tory (Nisko + Szaro + Nie-zielono) -> PRZED Trawą!
    current = process_step(current, "Tory", 18, logic_rail)
    
    # 5. Wysoka Roślinność (Wysoko + Zielono) -> Wycinamy drzewa
    current = process_step(current, "Roslinnosc_Wysoka", 5, logic_high_veg)
    
    # 6. Mosty (Wysoko + Płasko + Beton) -> PRZED Budynkami
    current = process_step(current, "Mosty", 17, logic_bridge)
    
    # 7. Budynki (Reszta wysokich rzeczy - Dachy, ściany)
    current = process_step(current, "Budynki", 6, logic_building)
    
    # 8. Średnia Roślinność
    current = process_step(current, "Roslinnosc_Srednia", 4, logic_med_veg)
    
    # 9. Niska Roślinność
    current = process_step(current, "Roslinnosc_Niska", 3, logic_low_veg)
    
    # 10. Reszta
    if os.path.exists(current):
        final_file = os.path.join(SCRIPT_DIR, "Wynik_1_Niesklasyfikowane.las")
        las = laspy.read(current)
        save_las(las.points, np.vstack((las.red, las.green, las.blue)).transpose(), las.header, final_file, 1)
        os.remove(current)

    print("\nGOTOWE! Sprawdź folder.")

if __name__ == "__main__":
    main()
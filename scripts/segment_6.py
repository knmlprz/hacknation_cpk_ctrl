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

# Globalny model terenu
GROUND_MODEL = None 

# ==========================================
# FUNKCJE POMOCNICZE (BEZ SĄSIEDZTWA)
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
    """Zwraca wysokość punktu nad modelem terenu"""
    [a, b, c, d] = GROUND_MODEL
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    # Jeśli wektor normalny skierowany w dół, odwracamy
    if c < 0: dists = -dists
    return dists

# ==========================================
# ZAAWANSOWANA ANALIZA KOLORU (VECTORIZED)
# ==========================================

def is_vegetation_color(rgb):
    """
    Wykrywa roślinność (Liściastą, Iglastą, Pnie).
    Działa na czystej matematyce RGB bez szukania sąsiadów.
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    # 1. Jasna zieleń (Trawa, Liście)
    # Green dominuje Red i Blue
    is_green = (g > r * 1.05) & (g > b * 0.95)
    
    # 2. Ciemna zieleń / Iglaki (Bardzo ciemne punkty, gdzie G jest relatywnie najsilniejsze)
    # Warunek: G >= R (lub blisko), G > B, ale całość jest ciemna (< 18000 w skali 16bit)
    is_dark_veg = (g >= r * 0.9) & (g > b) & (r < 18000)
    
    # 3. Brąz / Pnie / Suche gałęzie
    # Schemat brązu: R > G > B.
    # Unikamy jaskrawej czerwieni (samochody/dachy) sprawdzając czy R nie jest za duże względem G
    is_brown = (r > g) & (g > b) & (r < g * 2.0) & (r > 3000)
    
    return is_green | is_dark_veg | is_brown

def is_concrete_asphalt_color(rgb):
    """
    Wykrywa beton, asfalt, kamień (Mosty, Budynki).
    Szary = małe odchylenie standardowe między R,G,B.
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    # Odchylenie standardowe wzdłuż osi kolorów dla każdego punktu
    # Jeśli R, G i B są blisko siebie -> Szary/Biały/Czarny
    std_dev = np.std(rgb, axis=1)
    
    # Tolerancja szarości
    is_grey = std_dev < 5000 
    
    # Jasne elewacje (mogą być lekko kremowe, więc nie idealnie szare, ale bardzo jasne)
    brightness = np.mean(rgb, axis=1)
    is_bright = brightness > 25000
    
    return is_grey | is_bright

def is_rail_color(rgb):
    """Rdza i Ciemna Stal"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    std_dev = np.std(rgb, axis=1)
    # Ciemny szary/rdzawy, ale nie zielony
    return (std_dev < 4000) & (r < 20000) & (g < 20000) & (b < 20000)

def is_blue_water(rgb):
    """Woda"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # Wyraźnie niebieski LUB bardzo ciemny (czarna woda)
    return ((b > r) & (b > g)) | ((r<6000)&(g<6000)&(b<6000))

# ==========================================
# LOGIKI FILTRUJĄCE (PURE NUMPY)
# ==========================================

def logic_ground(xyz, rgb):
    global GROUND_MODEL
    # RANSAC tylko na małej próbce dla modelu - to nie obciąża RAMu
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
    return h < -2.0 # Głęboko pod ziemią

def logic_bridge(xyz, rgb):
    """
    MOSTY (Metoda bez sąsiedztwa):
    Wysoko + Kolor Betonu + Brak Roślinności.
    """
    h = get_heights(xyz)
    
    # Kryteria:
    # 1. Wysokość: Mosty są wysokie (>3.5m)
    # 2. Kolor: Beton/Asfalt
    # 3. Negacja: To NIE jest drzewo
    
    mask = (h > 3.5) & \
           is_concrete_asphalt_color(rgb) & \
           (~is_vegetation_color(rgb))
           
    return mask

def logic_water(xyz, rgb):
    h = get_heights(xyz)
    # Bardzo nisko i niebiesko/czarno
    return (h < 0.2) & is_blue_water(rgb)

def logic_rail(xyz, rgb):
    """TORY"""
    h = get_heights(xyz)
    # Nisko na nasypie, kolor stali/rdzy, nie zielony
    return (h > 0.1) & (h < 1.5) & is_rail_color(rgb) & (~is_vegetation_color(rgb))

def logic_high_veg(xyz, rgb):
    """DRZEWA"""
    h = get_heights(xyz)
    # Po prostu: Wysoko i ma kolor roślinny (zielony lub brązowy)
    return (h > 5.0) & is_vegetation_color(rgb)

def logic_buildings(xyz, rgb):
    """BUDYNKI (Reszta)"""
    h = get_heights(xyz)
    # Wszystko co zostało wysokiego (>2.5m).
    # Ponieważ mosty (szare) i drzewa (zielone/brązowe) już wycieliśmy,
    # to co zostało to dachy (często czerwone/czarne) i ściany.
    return (h > 2.5)

def logic_med_veg(xyz, rgb):
    h = get_heights(xyz)
    # Średnia wysokość + kolor rośliny
    return (h >= 1.5) & (h <= 5.0) & is_vegetation_color(rgb)

def logic_low_veg(xyz, rgb):
    h = get_heights(xyz)
    # Niska wysokość (ale nad wodą/torami)
    return (h > 0.2) & (h < 1.5) & is_vegetation_color(rgb)

# ==========================================
# SILNIK (Pipeline)
# ==========================================

def process_step(input_path, output_class_name, class_code, filter_logic_function):
    print(f"\n--- Filtrowanie: {output_class_name} (Kod {class_code}) ---")
    if not os.path.exists(input_path): return None

    # Czytanie
    try: las = laspy.read(input_path)
    except: return None

    # Szybka konwersja (Views - oszczędność pamięci, nie kopiujemy jeśli nie trzeba)
    # Ale do zaawansowanych masek numpy lepiej skopiować współrzędne
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    rgb = np.vstack((las.red, las.green, las.blue)).transpose()
    
    # 1. ZASTOSOWANIE LOGIKI (Szybka maska boolowska)
    mask = filter_logic_function(xyz, rgb)

    # 2. SEPARACJA
    class_points = las.points[mask]
    class_colors = rgb[mask]
    
    remainder_points = las.points[~mask]
    remainder_colors = rgb[~mask]
    
    print(f"   Wynik: {len(class_points)} pkt (Klasa) | {len(remainder_points)} pkt (Reszta)")

    # 3. ZAPIS
    class_file = os.path.join(SCRIPT_DIR, f"Wynik_{class_code}_{output_class_name}.las")
    save_las(class_points, class_colors, las.header, class_file, class_code)

    temp_out = os.path.join(SCRIPT_DIR, "temp_next_step.las")
    save_las(remainder_points, remainder_colors, las.header, temp_out, 1)

    # 4. CZYSZCZENIE RAM
    del las, xyz, rgb, mask, class_points, remainder_points
    gc.collect()

    # Rotacja plików
    if os.path.exists(input_path) and input_path != INPUT_FILE:
        os.remove(input_path)
    os.rename(temp_out, TEMP_FILE)
    
    return TEMP_FILE

def main():
    print("=== START KLASYFIKACJI (NO NEIGHBORS - PURE NUMPY) ===")
    
    current = INPUT_FILE
    
    # KOLEJNOŚĆ (Pipeline)
    
    # 1. Grunt (Musi być pierwszy)
    current = process_step(current, "Grunt", 2, logic_ground)
    
    # 2. Szum
    current = process_step(current, "Szum", 7, logic_noise)
    
    # 3. Woda (Bardzo specyficzna, łatwa do wycięcia)
    current = process_step(current, "Woda", 9, logic_water)
    
    # 4. Tory (Specyficzne, niskie)
    current = process_step(current, "Tory", 18, logic_rail)
    
    # 5. MOSTY (Wysokie, szare) - WAŻNE: Przed budynkami i drzewami
    current = process_step(current, "Mosty", 17, logic_bridge)
    
    # 6. Wysoka Roślinność (Wysoka, Zielona/Brązowa) - WAŻNE: Przed budynkami
    current = process_step(current, "Roslinnosc_Wysoka", 5, logic_high_veg)
    
    # 7. Budynki (To co zostało wysokiego, a nie jest mostem ani drzewem)
    current = process_step(current, "Budynki", 6, logic_buildings)
    
    # 8. Średnia Roślinność
    current = process_step(current, "Roslinnosc_Srednia", 4, logic_med_veg)
    
    # 9. Niska Roślinność
    current = process_step(current, "Roslinnosc_Niska", 3, logic_low_veg)
    
    # 10. Reszta -> Niesklasyfikowane
    if os.path.exists(current):
        final_file = os.path.join(SCRIPT_DIR, "Wynik_1_Niesklasyfikowane.las")
        las = laspy.read(current)
        save_las(las.points, np.vstack((las.red, las.green, las.blue)).transpose(), las.header, final_file, 1)
        os.remove(current)

    print("\nGOTOWE! Proces zakończony bez użycia metod sąsiedztwa.")

if __name__ == "__main__":
    main()
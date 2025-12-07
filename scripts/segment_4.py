import laspy
import numpy as np
import open3d as o3d
import os
import gc  # Garbage Collector do czyszczenia RAM

# --- KONFIGURACJA ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "Chmura_zadanie.las")
TEMP_FILE = os.path.join(SCRIPT_DIR, "temp_remainder.las")

# Parametry globalne
GROUND_THRESHOLD = 0.30     # Grubość warstwy gruntu
SAMPLE_SIZE = 100000        # Próbka do modelu terenu

# Zmienna globalna do przechowywania równania płaszczyzny ziemi (a, b, c, d)
# Jest potrzebna, żeby w kolejnych krokach wiedzieć co jest "wysoko" a co "nisko"
GROUND_MODEL = None 

def save_las(points, header_template, filename, classification_val):
    """Pomocnicza funkcja do zapisu pliku LAS z konkretną klasą"""
    if len(points) == 0:
        print(f"   [INFO] Brak punktów dla klasy {classification_val}. Pomijam zapis.")
        return

    out = laspy.LasData(header_template)
    out.points = points
    out.classification[:] = classification_val
    out.write(filename)
    print(f"   -> Zapisano: {os.path.basename(filename)} ({len(points)} pkt)")

def is_color_green(r, g, b):
    """Sprawdza czy kolor jest dominująco zielony (Roślinność)"""
    # Zielony musi być silniejszy niż czerwony i niebieski
    return (g > r) & (g > b)

def is_color_blue(r, g, b):
    """Sprawdza czy kolor jest dominująco niebieski (Woda)"""
    return (b > r) & (b > g * 1.2) # Niebieski dominujący

def is_color_dark(r, g, b, threshold=10000):
    """Sprawdza czy kolor jest ciemny (Woda/Asfalt) - dla skali 16bit"""
    return (r < threshold) & (g < threshold) & (b < threshold)

def process_step(input_path, output_class_name, class_code, filter_logic_function):
    """
    GENERYCZNA FUNKCJA KROKU:
    1. Wczytaj aktualny plik resztek.
    2. Zastosuj logikę filtrującą.
    3. Zapisz wyciętą klasę.
    4. Zapisz nową resztę do pliku tymczasowego.
    5. Wyczyść RAM.
    """
    print(f"\n--- Przetwarzanie: {output_class_name} (Kod {class_code}) ---")
    
    if not os.path.exists(input_path):
        print("Błąd: Brak pliku wejściowego dla tego kroku!")
        return None

    try:
        las = laspy.read(input_path)
    except Exception as e:
        print(f"Błąd odczytu: {e}")
        return None

    # Konwersja do numpy (potrzebne do obliczeń)
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    rgb = np.vstack((las.red, las.green, las.blue)).transpose()
    
    # Wywołanie logiki filtrującej (zwraca maskę True/False)
    # True = Punkt należy do szukanej klasy
    # False = Punkt należy do reszty
    mask = filter_logic_function(xyz, rgb, las)

    # Podział punktów
    class_points = las.points[mask]
    remainder_points = las.points[~mask]
    
    count_class = len(class_points)
    count_rem = len(remainder_points)
    print(f"   Znaleziono: {count_class} | Pozostało: {count_rem}")

    # 1. Zapisz plik z klasą
    class_file = os.path.join(SCRIPT_DIR, f"Wynik_{class_code}_{output_class_name}.las")
    save_las(class_points, las.header, class_file, class_code)

    # 2. Zapisz resztę do NOWEGO pliku tymczasowego (nadpisz stary)
    # Musimy to zrobić, żeby zwolnić RAM przed następnym krokiem
    temp_out = os.path.join(SCRIPT_DIR, "temp_next_step.las")
    save_las(remainder_points, las.header, temp_out, 1) # Tymczasowo jako 1

    # 3. Sprzątanie
    del las, xyz, rgb, mask, class_points, remainder_points
    gc.collect()

    # Zamiana plików temp (nowy staje się obecnym)
    if os.path.exists(input_path) and input_path != INPUT_FILE:
        os.remove(input_path) # Usuń stary temp
    
    os.rename(temp_out, TEMP_FILE) # Nowy temp staje się głównym tempem
    
    return TEMP_FILE

# --- LOGIKI FILTRUJĄCE ---

def logic_ground(xyz, rgb, las_obj):
    """Logika RANSAC dla Gruntu"""
    global GROUND_MODEL
    
    # Próbka dla szybkości
    if len(xyz) > SAMPLE_SIZE:
        idx = np.random.choice(len(xyz), SAMPLE_SIZE, replace=False)
        sample = xyz[idx]
    else:
        sample = xyz

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample)
    
    # RANSAC
    model, _ = pcd.segment_plane(distance_threshold=GROUND_THRESHOLD, ransac_n=3, num_iterations=1000)
    GROUND_MODEL = model # Zapisujemy model globalnie (a,b,c,d)
    [a, b, c, d] = model
    print(f"   Model terenu ustalony: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Obliczamy odległość wszystkich punktów od płaszczyzny
    distances = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    
    # Upewniamy się, że "góra" to wartości dodatnie (wektor normalny w górę)
    if c < 0: distances = -distances
    
    # Zapisujemy odległość od gruntu jako nowy wymiar w obiekcie las (trik do przekazywania danych)
    # Niestety laspy object jest read-only w tej strukturze funkcji, więc obliczymy to w locie
    # w kolejnych krokach, używając GROUND_MODEL.
    
    # Grunt to punkty blisko zera
    return np.abs(distances) < GROUND_THRESHOLD

def get_heights(xyz):
    """Pomocnicza: Liczy wysokość nad gruntem na podstawie zapisanego modelu"""
    [a, b, c, d] = GROUND_MODEL
    dists = (xyz[:, 0]*a + xyz[:, 1]*b + xyz[:, 2]*c + d) / np.sqrt(a**2+b**2+c**2)
    if c < 0: dists = -dists
    return dists

def logic_noise(xyz, rgb, las_obj):
    """Szum: punkty głęboko pod ziemią"""
    h = get_heights(xyz)
    return h < -1.0  # Wszystko metr pod ziemią to błąd

def logic_water(xyz, rgb, las_obj):
    """Woda: Nisko, Płasko i Niebiesko/Ciemno"""
    h = get_heights(xyz)
    
    # Warunek 1: Wysokość (bardzo nisko, np. w zagłębieniach terenu)
    # Zakładamy, że woda jest poniżej poziomu "gruntu 0" lub tuż nad nim
    is_low = h < 0.2
    
    # Warunek 2: Kolor
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    is_blue_or_dark = is_color_blue(r, g, b) | is_color_dark(r, g, b)
    
    return is_low & is_blue_or_dark

def logic_high_veg(xyz, rgb, las_obj):
    """Wysoka roślinność: Wysoko + Zielono LUB Wysoko + Chaotycznie"""
    h = get_heights(xyz)
    
    # Warunek 1: Wysokość > 5m
    is_high = h > 5.0
    
    # Warunek 2: Kolor Zielony
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    is_green = is_color_green(r, g, b)
    
    # Można by tu dodać analizę geometrii (Scattering), ale to spowalnia.
    # Uproszczenie: Wszystko co wysokie i zielone to drzewo.
    # A co jeśli nie jest zielone (zimą)? Wtedy wszystko co wysokie i nie jest płaskim budynkiem.
    # Ale budynki wytniemy później, więc na razie bierzemy to co pewne (zielone).
    
    return is_high & is_green

def logic_buildings(xyz, rgb, las_obj):
    """Budynki: To co zostało wysokiego i nie jest zielone (bo drzewa wycięliśmy krok wcześniej)"""
    h = get_heights(xyz)
    
    # Budynek > 2.5m (żeby nie łapać płotów/samochodów)
    is_high = h > 2.5
    
    # Dodatkowa weryfikacja: Płaskość (Planarity) lub Pionowość (Verticality)
    # Tu użyjemy prostego RANSACa ponownie, żeby znaleźć ściany/dachy w tym co zostało
    # Ale dla uproszczenia (bo drzewa już zniknęły w 90%): 
    # Bierzemy wszystko co wysokie.
    
    return is_high

def logic_med_veg(xyz, rgb, las_obj):
    """Średnia roślinność: 1.5m - 5m + Zielono"""
    h = get_heights(xyz)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    is_med_height = (h >= 1.5) & (h <= 5.0)
    # Bierzemy wszystko w tej wysokości, co zostało (bo budynki wycięliśmy)
    # Lub doprecyzowujemy kolorem
    return is_med_height

def logic_low_veg(xyz, rgb, las_obj):
    """Niska roślinność: 0.3m - 1.5m"""
    h = get_heights(xyz)
    is_low_height = (h > 0.3) & (h < 1.5)
    return is_low_height

# --- GŁÓWNA PĘTLA ---

def main():
    print("=== START KASKADOWEJ KLASYFIKACJI ===")
    
    # KROK 1: GRUNT (Inicjacja z pliku oryginalnego)
    # Wynik: Plik Gruntu + Temp_Remainder
    current_input = INPUT_FILE
    current_input = process_step(current_input, "Grunt", 2, logic_ground)
    
    if not current_input: return # Błąd krytyczny

    # Od teraz current_input to zawsze "temp_remainder.las"
    
    # KROK 2: SZUM (Głęboko pod ziemią)
    current_input = process_step(current_input, "Szum", 7, logic_noise)

    # KROK 3: WODA (Nisko i niebiesko)
    current_input = process_step(current_input, "Woda", 9, logic_water)
    
    # KROK 4: WYSOKA ROŚLINNOŚĆ (Drzewa > 5m, Zielone)
    current_input = process_step(current_input, "Roslinnosc_Wysoka", 5, logic_high_veg)
    
    # KROK 5: BUDYNKI (To co zostało wysokiego, a nie jest zielonym drzewem)
    current_input = process_step(current_input, "Budynki", 6, logic_buildings)
    
    # KROK 6: ŚREDNIA ROŚLINNOŚĆ
    current_input = process_step(current_input, "Roslinnosc_Srednia", 4, logic_med_veg)
    
    # KROK 7: NISKA ROŚLINNOŚĆ
    current_input = process_step(current_input, "Roslinnosc_Niska", 3, logic_low_veg)
    
    # KROK 8: RESZTA (Unclassified)
    # To co zostało w pliku temp, zmieniamy nazwę na wynikową
    if os.path.exists(current_input):
        final_unclassified = os.path.join(SCRIPT_DIR, "Wynik_1_Niesklasyfikowane.las")
        
        # Wczytaj żeby zmienić klasę na 1 i zapisać porządnie
        las = laspy.read(current_input)
        save_las(las.points, las.header, final_unclassified, 1)
        os.remove(current_input) # Usuń temp
        print(f"\n--- Koniec. Reszta zapisana jako: {os.path.basename(final_unclassified)} ---")

    print("\nGOTOWE! Wszystkie klasy wyodrębnione sekwencyjnie.")

if __name__ == "__main__":
    main()
import laspy
import numpy as np
import open3d as o3d
import os

# --- KONFIGURACJA ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "Chmura_zadanie.las")

# Nazwy plików wyjściowych
OUTPUT_GROUND = os.path.join(SCRIPT_DIR, "Wynik_grunt.las")
OUTPUT_OBJECTS = os.path.join(SCRIPT_DIR, "Wynik_obiekty.las")

# Parametry
DISTANCE_THRESHOLD = 0.30  # Tolerancja (30cm)
SAMPLE_SIZE = 100000       # Próbka do obliczeń (nie wpływa na jakość zapisu)

def main():
    print(f"1. Wczytywanie: {INPUT_FILE}...")
    try:
        las = laspy.read(INPUT_FILE)
    except FileNotFoundError:
        print("Nie znaleziono pliku.")
        return

    # Pobieramy punkty do obliczeń
    points_xyz = np.vstack((las.x, las.y, las.z)).transpose()
    total_points = len(points_xyz)
    print(f"   Liczba punktów: {total_points}")

    # --- KROK 2: Obliczanie modelu (RANSAC) ---
    print("2. Analiza geometrii...")
    
    # Próbka dla szybkości
    if total_points > SAMPLE_SIZE:
        sample_indices = np.random.choice(total_points, SAMPLE_SIZE, replace=False)
        sample_points = points_xyz[sample_indices]
    else:
        sample_points = points_xyz

    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(sample_points)

    plane_model, _ = pcd_sample.segment_plane(distance_threshold=DISTANCE_THRESHOLD,
                                              ransac_n=3,
                                              num_iterations=1000)
    
    [a, b, c, d] = plane_model
    
    # Aplikacja modelu do całej chmury
    distances = np.abs(points_xyz[:, 0] * a + 
                       points_xyz[:, 1] * b + 
                       points_xyz[:, 2] * c + d)

    # Maska logiczna (True = Grunt, False = Obiekt)
    ground_mask = distances < DISTANCE_THRESHOLD
    
    count_ground = np.sum(ground_mask)
    count_objects = total_points - count_ground
    
    print(f"   Podział: {count_ground} (grunt) vs {count_objects} (obiekty)")

    # --- KROK 3: Zapisywanie oddzielnych plików (BEZ ZMIANY KOLORÓW) ---
    
    # A. ZAPIS GRUNTU
    print(f"3. Zapisywanie pliku GRUNTU: {OUTPUT_GROUND}...")
    # Tworzymy widok (slice) oryginalnego pliku - zachowuje to kolory i atrybuty
    ground_las = las[ground_mask]
    ground_las.write(OUTPUT_GROUND)

    # B. ZAPIS OBIEKTÓW
    print(f"4. Zapisywanie pliku OBIEKTÓW: {OUTPUT_OBJECTS}...")
    # To samo dla reszty
    objects_las = las[~ground_mask]
    objects_las.write(OUTPUT_OBJECTS)

    print("Gotowe! Masz teraz dwa pliki z oryginalnymi kolorami.")

if __name__ == "__main__":
    main()
import laspy
import numpy as np
import open3d as o3d
import copy

# KONFIGURACJA
INPUT_FILE = "Chmura_zadanie.las"  # Twój plik wejściowy (musi być w tym samym folderze)
OUTPUT_FILE = "Chmura_posegmentowana.las"
DISTANCE_THRESHOLD = 0.30  # Tolerancja dla gruntu (w metrach, np. 30cm)

def main():
    print(f"1. Wczytywanie pliku: {INPUT_FILE}...")
    try:
        las = laspy.read(INPUT_FILE)
    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku. Sprawdź nazwę i ścieżkę.")
        return

    # Pobranie współrzędnych XYZ do formatu numpy
    points = np.stack([las.x, las.y, las.z], axis=1)
    print(f"   Liczba punktów: {len(points)}")

    # Konwersja do formatu Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print("2. Przetwarzanie: Segmentacja płaszczyzny (RANSAC)...")
    # segment_plane zwraca model matematyczny płaszczyzny i indeksy punktów, które do niej należą
    plane_model, inliers = pcd.segment_plane(distance_threshold=DISTANCE_THRESHOLD,
                                             ransac_n=3,
                                             num_iterations=1000)
    
    #  - tag dla kontekstu (zgodnie z instrukcją nie wyświetli się obrazek, ale to jest miejsce gdzie algorytm działa)

    print(f"   Znaleziono {len(inliers)} punktów gruntu.")

    # 3. Modyfikacja danych w pliku LAS
    # Tworzymy maskę logiczną (True dla gruntu, False dla reszty)
    is_ground = np.zeros(len(points), dtype=bool)
    is_ground[inliers] = True

    # --- Aktualizacja Klasyfikacji (Standard LAS) ---
    # Jeśli plik nie ma pola classification, laspy je obsłuży automatycznie przy zapisie nowszych wersji,
    # ale zazwyczaj ono istnieje.
    
    # Ustawiamy wszystko jako "Unclassified" (1)
    las.classification[:] = 1
    
    # Ustawiamy inliers jako "Ground" (2)
    las.classification[is_ground] = 2

    # --- Aktualizacja Kolorów (Opcjonalne - dla lepszej wizualizacji) ---
    # Format LAS używa 16-bitowych kolorów (0-65535)
    # Czerwony dla obiektów, Zielony dla gruntu
    
    # Tworzymy nowe tablice kolorów
    red = np.zeros(len(points), dtype=np.uint16)
    green = np.zeros(len(points), dtype=np.uint16)
    blue = np.zeros(len(points), dtype=np.uint16)

    # Obiekty (Reszta) -> Czerwony
    red[~is_ground] = 65535 
    
    # Grunt -> Zielony
    green[is_ground] = 65535

    # Przypisanie do obiektu las
    las.red = red
    las.green = green
    las.blue = blue

    print(f"3. Zapisywanie wyniku do: {OUTPUT_FILE}...")
    las.write(OUTPUT_FILE)
    print("Gotowe! Możesz otworzyć plik w CloudCompare.")

if __name__ == "__main__":
    main()
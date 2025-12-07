import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import subprocess
import threading
import sys
import os

# --- KONFIGURACJA ---
# ZMIANA: Wskazujemy na Twój plik z logiką segmentacji
WORKER_SCRIPT = "segment_8.py" 

# --- KOLORYSTYKA ---
COLOR_BG = "#FFFFFF"
COLOR_TEAL = "#264653"
COLOR_RED = "#E63946"
COLOR_TEXT = "#1D3557"
COLOR_BTN_GREY = "#6C757D"
COLOR_BTN_HOVER = "#495057"
COLOR_CONSOLE_BG = "#000000"
COLOR_CONSOLE_FG = "#00FF00"

# Fonty
FONT_HEADER = ("Arial Black", 24, "bold")
FONT_LABEL = ("Segoe UI", 9, "bold")
FONT_ENTRY = ("Consolas", 9)

# --- KLASA PRZYCISKU ---
class ResponsiveRoundedButton(tk.Canvas):
    def __init__(self, parent, height, corner_radius, color, hover_color, text, text_color, command):
        super().__init__(parent, borderwidth=0, relief="flat", highlightthickness=0, bg=parent["bg"])
        self.command = command
        self.color = color
        self.hover_color = hover_color
        self.height = height
        self.corner_radius = corner_radius
        self.text_content = text
        self.text_color = text_color
        
        self.configure(height=height)
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Configure>", self.on_resize)
        
        self.rect_id = None
        self.text_id = None

    def on_resize(self, event):
        w = event.width
        h = self.height
        self.delete("all")
        self.rect_id = self.round_rect(0, 0, w, h, self.corner_radius, fill=self.color, outline="")
        self.text_id = self.create_text(w/2, h/2, text=self.text_content, fill=self.text_color, font=("Segoe UI", 9, "bold"))

    def round_rect(self, x, y, w, h, r, **kwargs):
        points = (x+r, y, x+r, y, x+w-r, y, x+w-r, y, x+w, y, x+w, y+r, x+w, y+r, x+w, y+h-r, x+w, y+h-r, x+w, y+h, x+w-r, y+h, x+w-r, y+h, x+r, y+h, x+r, y+h, x, y+h, x, y+h-r, x, y+h-r, x, y+r, x, y+r, x, y)
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_enter(self, event):
        self.itemconfig(self.rect_id, fill=self.hover_color)
    def on_leave(self, event):
        self.itemconfig(self.rect_id, fill=self.color)
    def on_click(self, event):
        self.move(self.text_id, 1, 1); self.move(self.rect_id, 0, 0)
    def on_release(self, event):
        self.move(self.text_id, -1, -1)
        if self.command: self.command()
    def set_text(self, text):
        self.itemconfig(self.text_id, text=text)
    def disable(self):
        self.unbind("<Button-1>"); self.unbind("<Enter>")
        self.itemconfig(self.rect_id, fill="#CCCCCC")
    def enable(self):
        self.bind("<Button-1>", self.on_click); self.bind("<Enter>", self.on_enter)
        self.itemconfig(self.rect_id, fill=self.color)


class HacknationSimpleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SEGMENTACJA: CLOUDSHAPE")
        
        self.root.geometry("600x600")
        self.root.minsize(550, 500)
        self.root.configure(bg=COLOR_BG)

        # --- NAGŁÓWEK ---
        header_frame = tk.Frame(root, bg=COLOR_BG, pady=20)
        header_frame.pack(fill="x")
        center_box = tk.Frame(header_frame, bg=COLOR_BG)
        center_box.pack(anchor="center")
        tk.Label(center_box, text="SEGMENTACJA ", font=FONT_HEADER, fg=COLOR_TEAL, bg=COLOR_BG).pack(side="left")
        tk.Label(center_box, text="CLOUDSHAPE", font=FONT_HEADER, fg=COLOR_RED, bg=COLOR_BG).pack(side="left")

        # --- OBSZAR PRZEWIJANY ---
        self.canvas = tk.Canvas(root, bg=COLOR_BG, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLOR_BG, padx=20, pady=20)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=550)
        self.canvas.bind("<Configure>", self._configure_inner_frame_width)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_linux_scroll_up)
        self.canvas.bind_all("<Button-5>", self._on_linux_scroll_down)

        # --- SEKCJA GŁÓWNA ---
        frame = tk.Frame(self.scrollable_frame, bg=COLOR_BG)
        frame.pack(fill="x", pady=10)
        
        tk.Label(frame, text="PLIK WEJŚCIOWY (.las):", font=FONT_LABEL, bg=COLOR_BG, fg=COLOR_TEXT).pack(anchor="w")
        self.entry_input = tk.Entry(frame, font=FONT_ENTRY, relief="flat", bg="#F1F3F5", fg="#333")
        self.entry_input.config(highlightbackground=COLOR_TEAL, highlightthickness=1, bd=3)
        self.entry_input.pack(fill="x", ipady=5, pady=(2, 5))
        
        btn = ResponsiveRoundedButton(frame, height=35, corner_radius=15,
                                      color=COLOR_BTN_GREY, hover_color=COLOR_BTN_HOVER,
                                      text="WYBIERZ PLIK 📂", text_color="white", command=self.browse_input)
        btn.pack(fill="x")

        info_frame = tk.Frame(self.scrollable_frame, bg="#E8F4F8", padx=10, pady=10)
        info_frame.pack(fill="x", pady=20)
        tk.Label(info_frame, text="ℹ System automatycznie wygeneruje pliki klasyfikacyjne w folderze źródłowym.", 
                 bg="#E8F4F8", fg=COLOR_TEAL, font=("Segoe UI", 9), justify="left").pack(anchor="w")

        # 2. START
        self.btn_start = ResponsiveRoundedButton(self.scrollable_frame, height=55, corner_radius=27,
                                                 color=COLOR_RED, hover_color="#C01B29",
                                                 text="ROZPOCZNIJ ANALIZĘ ▶", text_color="white",
                                                 command=self.start_process)
        self.btn_start.pack(fill="x", pady=10)

        # 3. KONSOLA
        tk.Label(self.scrollable_frame, text="LOGI SYSTEMOWE:", font=("Segoe UI", 8, "bold"), bg=COLOR_BG, fg="#999").pack(anchor="w", pady=(20, 5))
        console_frame = tk.Frame(self.scrollable_frame, bg="black", bd=0)
        console_frame.pack(fill="x", pady=(0, 20))
        
        self.log_area = scrolledtext.ScrolledText(console_frame, state='disabled', height=12,
                                                  bg=COLOR_CONSOLE_BG, fg=COLOR_CONSOLE_FG,
                                                  font=("Consolas", 8), borderwidth=0)
        self.log_area.pack(fill="both", expand=True, padx=5, pady=5)

    def _configure_inner_frame_width(self, event):
        self.canvas.itemconfig(self.canvas_window_id, width=event.width)
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    def _on_linux_scroll_up(self, event):
        self.canvas.yview_scroll(-1, "units")
    def _on_linux_scroll_down(self, event):
        self.canvas.yview_scroll(1, "units")

    def log(self, txt):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, txt + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Pliki LAS", "*.las"), ("Wszystkie", "*.*")])
        if filename:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, filename)

    def start_process(self):
        input_path = self.entry_input.get()
        if not input_path:
            messagebox.showwarning("Błąd", "Wybierz plik wejściowy!")
            return

        if not os.path.exists(input_path):
            messagebox.showerror("Błąd", "Podany plik nie istnieje.")
            return

        base_name = os.path.splitext(input_path)[0]
        args_list = ["--input", input_path]
        
        output_map = {
            "ground": "_GRUNT",
            "noise": "_SZUM",
            "water": "_WODA",
            "rails": "_TORY",
            "bridges": "_MOSTY",
            "veg_high": "_VEG_HIGH",
            "buildings": "_BUDYNKI",
            "veg_med": "_VEG_MED",
            "veg_low": "_VEG_LOW",
            "unclassified": "_RESZTA"
        }

        self.log(f">>> Przygotowywanie plików wyjściowych...")
        for arg_name, suffix in output_map.items():
            out_path = f"{base_name}{suffix}.las"
            args_list.extend([f"--out_{arg_name}", out_path])

        self.btn_start.disable()
        self.btn_start.set_text("PRZETWARZANIE...")
        self.log_area.config(state='normal'); self.log_area.delete(1.0, tk.END); self.log_area.config(state='disabled')
        
        threading.Thread(target=self.run_worker, args=(args_list,), daemon=True).start()

    def run_worker(self, args_list):
        cmd = [sys.executable, WORKER_SCRIPT] + args_list
        try:
            self.log(f">>> START SILNIKA OBLICZENIOWEGO...")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            for line in iter(proc.stdout.readline, ''):
                self.log(line.strip())
            
            proc.stdout.close()
            ret = proc.wait()
            
            if ret == 0: 
                self.log("\n[SUKCES] ZAKOŃCZONO.")
                messagebox.showinfo("Sukces", "Segmentacja zakończona pomyślnie.")
            else: 
                self.log(f"\n[BŁĄD] Kod: {ret}")
                messagebox.showerror("Błąd", "Błąd procesu.")
        except Exception as e:
            self.log(f"[CRITICAL ERROR] {e}")
        finally: 
            self.root.after(0, lambda: self.btn_start.enable())
            self.root.after(0, lambda: self.btn_start.set_text("ROZPOCZNIJ ANALIZĘ ▶"))

if __name__ == "__main__":
    root = tk.Tk()
    app = HacknationSimpleApp(root)
    root.mainloop()
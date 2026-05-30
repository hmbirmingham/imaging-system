#!/usr/bin/env python3
"""
Plate Imaging System — dark-themed GUI with live camera controls,
scrollable capture gallery, and colony quantification via quantify.py.
"""

import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("picamera2 not available — running in demo mode")

from quantify import quantify_colonies
from anomaly import AnomalyDetector
from data_logger import DataLogger

# Initialise shared detector and logger (load saved model if present)
_detector = AnomalyDetector()
_logger   = DataLogger()

# ── Directories ───────────────────────────────────────────────────────────────
SAVE_DIR   = Path("captures")
RESULT_DIR = Path("results")
SAVE_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# ── Layout constants ──────────────────────────────────────────────────────────
PREVIEW_W, PREVIEW_H = 500, 500
THUMB_SIZE = 90

# ── Colour palette (Catppuccin Mocha) ────────────────────────────────────────
C = {
    "bg":      "#1e1e2e",
    "surface": "#2a2a3e",
    "overlay": "#313244",
    "border":  "#45475a",
    "text":    "#cdd6f4",
    "subtext": "#a6adc8",
    "accent":  "#89b4fa",
    "green":   "#a6e3a1",
    "yellow":  "#f9e2af",
    "red":     "#f38ba8",
    "mantle":  "#181825",
}


class PlateImagingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Plate Imaging System")
        self.root.configure(bg=C["bg"])
        self.root.minsize(900, 600)

        self.previewing = True
        self.selected_capture: Optional[Path] = None
        self._preview_ref = None
        self._thumb_refs: list = []

        self._setup_camera()
        self._setup_styles()
        self._build_ui()

        if CAMERA_AVAILABLE:
            self.root.after(33, self._update_frame)
        else:
            self._show_no_camera()

    # ── Camera setup ──────────────────────────────────────────────────────────

    def _setup_camera(self) -> None:
        global CAMERA_AVAILABLE
        if not CAMERA_AVAILABLE:
            self.cam = None
            return
        try:
            self.cam = Picamera2()
            # OV9281 uses 640x400; IMX477 and most others support 640x480
            # Try 640x480 first, fall back to 640x400 if unsupported
            try:
                cfg = self.cam.create_preview_configuration(main={"size": (640, 480)})
                self.cam.configure(cfg)
            except Exception:
                cfg = self.cam.create_preview_configuration(main={"size": (640, 400)})
                self.cam.configure(cfg)
            self.cam.start()
        except Exception as e:
            print(f"Camera init failed: {e}")
            self.cam = None
            CAMERA_AVAILABLE = False

    # ── ttk styles ────────────────────────────────────────────────────────────

    def _setup_styles(self) -> None:
        s = ttk.Style(self.root)
        s.theme_use("clam")

        # Base
        s.configure(".",
                    background=C["bg"], foreground=C["text"],
                    fieldbackground=C["surface"], bordercolor=C["border"],
                    troughcolor=C["surface"], relief="flat",
                    font=("Helvetica", 11))

        # Frames
        s.configure("TFrame",          background=C["bg"])
        s.configure("Surface.TFrame",  background=C["surface"])
        s.configure("Mantle.TFrame",   background=C["mantle"])

        # Labels
        s.configure("TLabel",          background=C["bg"],      foreground=C["text"])
        s.configure("Surface.TLabel",  background=C["surface"], foreground=C["text"])
        s.configure("Subtext.TLabel",  background=C["bg"],      foreground=C["subtext"],
                    font=("Helvetica", 9))
        s.configure("SurfaceSub.TLabel", background=C["surface"], foreground=C["subtext"],
                    font=("Helvetica", 9))
        s.configure("Header.TLabel",   background=C["bg"],      foreground=C["accent"],
                    font=("Helvetica", 12, "bold"))
        s.configure("SurfaceHdr.TLabel", background=C["surface"], foreground=C["accent"],
                    font=("Helvetica", 12, "bold"))
        s.configure("Accent.TLabel",   background=C["surface"], foreground=C["accent"],
                    font=("Helvetica", 10))
        s.configure("Status.TLabel",   background=C["mantle"],  foreground=C["subtext"],
                    font=("Helvetica", 9), padding=(8, 3))

        # Buttons
        s.configure("TButton",
                    background=C["surface"], foreground=C["text"],
                    bordercolor=C["border"], padding=(10, 5), focusthickness=0)
        s.map("TButton",
              background=[("active", C["overlay"]), ("pressed", C["mantle"])],
              foreground=[("active", C["accent"])])

        s.configure("Capture.TButton",
                    background=C["accent"], foreground=C["mantle"],
                    font=("Helvetica", 13, "bold"), padding=(18, 9))
        s.map("Capture.TButton",
              background=[("active", "#74c7ec"), ("pressed", "#59a6d4")])

        s.configure("Quant.TButton",
                    background=C["green"], foreground=C["mantle"],
                    font=("Helvetica", 12, "bold"), padding=(14, 7))
        s.map("Quant.TButton",
              background=[("active", "#8ece9a"), ("pressed", "#6eb882")],
              foreground=[("disabled", C["border"])])

        s.configure("Warn.TButton",
                    background=C["overlay"], foreground=C["subtext"],
                    font=("Helvetica", 10), padding=(10, 5))
        s.map("Warn.TButton",
              background=[("active", C["border"])])

        # Slider
        s.configure("TScale",
                    background=C["bg"], troughcolor=C["surface"],
                    sliderlength=16, sliderrelief="flat")

        s.configure("TSeparator", background=C["border"])

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()
        self._build_status_bar()

    # Left: preview canvas
    def _build_left_panel(self) -> None:
        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Title row
        hdr = ttk.Frame(left)
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(hdr, text="Live Preview", style="Header.TLabel").pack(side="left")
        self.preview_mode_label = ttk.Label(hdr, text="● LIVE", style="Subtext.TLabel")
        self.preview_mode_label.pack(side="left", padx=(10, 0))

        # Preview canvas
        self.preview_label = tk.Label(
            left,
            bg=C["mantle"],
            width=PREVIEW_W,
            height=PREVIEW_H,
            relief="flat",
            cursor="crosshair",
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        self.preview_label.bind("<Double-Button-1>", lambda _: self._resume_preview())

        ttk.Label(left, text="Double-click preview to return to live view",
                  style="Subtext.TLabel").grid(row=2, column=0, sticky="w", pady=(4, 0))

    # Right: controls + gallery + analysis
    def _build_right_panel(self) -> None:
        right = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)   # gallery stretches
        right.rowconfigure(7, weight=2)   # results stretches more

        # ── Camera controls card ──────────────────────────────────────
        card = ttk.Frame(right, style="Surface.TFrame", padding=12)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Camera Controls",
                  style="SurfaceHdr.TLabel").grid(row=0, column=0, columnspan=3,
                                                  sticky="w", pady=(0, 10))

        self._slider_row(card, row=1, label="Brightness",
                         from_=-1.0, to=1.0, init=0.0,
                         fmt="{:+.2f}", attr="brightness_var",
                         val_attr="brightness_lbl",
                         command=self._on_brightness)

        self._slider_row(card, row=2, label="Contrast",
                         from_=0.0, to=4.0, init=1.0,
                         fmt="{:.2f}", attr="contrast_var",
                         val_attr="contrast_lbl",
                         command=self._on_contrast)

        self._slider_row(card, row=3, label="Saturation",
                         from_=0.0, to=4.0, init=1.0,
                         fmt="{:.2f}", attr="saturation_var",
                         val_attr="saturation_lbl",
                         command=self._on_saturation)

        # ── Capture button ────────────────────────────────────────────
        self.capture_btn = ttk.Button(
            right, text="  Capture Image",
            style="Capture.TButton",
            command=self.capture_image,
            state="normal" if CAMERA_AVAILABLE else "disabled",
        )
        self.capture_btn.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ttk.Separator(right, orient="horizontal").grid(row=2, column=0,
                                                       sticky="ew", pady=(0, 8))

        # ── Gallery ───────────────────────────────────────────────────
        ttk.Label(right, text="Recent Captures",
                  style="Header.TLabel").grid(row=3, column=0, sticky="w", pady=(0, 6))

        gallery_outer = ttk.Frame(right, style="Surface.TFrame")
        gallery_outer.grid(row=4, column=0, sticky="ew", pady=(0, 8))

        self.gallery_canvas = tk.Canvas(
            gallery_outer, bg=C["surface"],
            height=THUMB_SIZE + 28, highlightthickness=0, bd=0,
        )
        hscroll = ttk.Scrollbar(gallery_outer, orient="horizontal",
                                command=self.gallery_canvas.xview)
        self.gallery_canvas.configure(xscrollcommand=hscroll.set)
        hscroll.pack(side="bottom", fill="x")
        self.gallery_canvas.pack(side="top", fill="both", expand=True)

        self.gallery_inner = tk.Frame(self.gallery_canvas, bg=C["surface"])
        self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")
        self.gallery_inner.bind("<Configure>",
            lambda e: self.gallery_canvas.configure(
                scrollregion=self.gallery_canvas.bbox("all")))

        ttk.Separator(right, orient="horizontal").grid(row=5, column=0,
                                                       sticky="ew", pady=(0, 8))

        # ── Analysis section ──────────────────────────────────────────
        ttk.Label(right, text="Colony Analysis",
                  style="Header.TLabel").grid(row=6, column=0, sticky="w", pady=(0, 6))

        self.quantify_btn = ttk.Button(
            right,
            text="Quantify Selected Image",
            style="Quant.TButton",
            command=self.quantify_selected,
            state="disabled",
        )
        self.quantify_btn.grid(row=7, column=0, sticky="ew", pady=(0, 8))

        # Results text box
        results_frame = ttk.Frame(right, style="Surface.TFrame", padding=8)
        results_frame.grid(row=8, column=0, sticky="nsew")
        right.rowconfigure(8, weight=3)

        self.results_text = tk.Text(
            results_frame,
            bg=C["surface"], fg=C["text"],
            font=("Courier", 10),
            relief="flat", state="disabled",
            wrap="word",
            insertbackground=C["accent"],
            selectbackground=C["border"],
            height=12,
        )
        results_scroll = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        results_scroll.pack(side="right", fill="y")
        self.results_text.pack(fill="both", expand=True)

        self._write_results(
            "No analysis yet.\n\n"
            "1. Capture an image\n"
            "2. Click \"Quantify Selected Image\"\n"
            "3. Results appear here"
        )

    def _build_status_bar(self) -> None:
        self.status_var = tk.StringVar(value="  Ready")
        ttk.Label(self.root, textvariable=self.status_var,
                  style="Status.TLabel",
                  anchor="w").grid(row=1, column=0, columnspan=2, sticky="ew")

    def _slider_row(self, parent, row, label, from_, to, init,
                    fmt, attr, val_attr, command) -> None:
        var = tk.DoubleVar(value=init)
        setattr(self, attr, var)

        ttk.Label(parent, text=label,
                  style="SurfaceSub.TLabel").grid(row=row, column=0, sticky="w",
                                                  padx=(0, 8), pady=(6, 0))
        ttk.Scale(parent, from_=from_, to=to, variable=var,
                  orient="horizontal",
                  command=lambda v, f=fmt, va=val_attr: self._update_slider_label(v, f, va)
                  ).grid(row=row, column=1, sticky="ew", pady=(6, 0))
        # patch: also wire the real callback
        var.trace_add("write", lambda *_, c=command, v=var: c(v.get()))

        lbl = ttk.Label(parent, text=fmt.format(init),
                        style="Accent.TLabel", width=6)
        lbl.grid(row=row, column=2, padx=(6, 0), pady=(6, 0))
        setattr(self, val_attr, lbl)

    def _update_slider_label(self, val, fmt, attr) -> None:
        getattr(self, attr).config(text=fmt.format(float(val)))

    # ── Camera control callbacks ──────────────────────────────────────────────

    def _on_brightness(self, val: float) -> None:
        if self.cam:
            try:
                self.cam.set_controls({"Brightness": float(val)})
            except Exception:
                pass

    def _on_contrast(self, val: float) -> None:
        if self.cam:
            try:
                self.cam.set_controls({"Contrast": float(val)})
            except Exception:
                pass

    def _on_saturation(self, val: float) -> None:
        if self.cam:
            try:
                self.cam.set_controls({"Saturation": float(val)})
            except Exception:
                pass

    # ── Preview loop ──────────────────────────────────────────────────────────

    def _update_frame(self) -> None:
        if self.previewing and self.cam:
            try:
                frame = self.cam.capture_array()
                img = Image.fromarray(frame)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                img = img.resize((PREVIEW_W, PREVIEW_H), Image.BILINEAR)
                img_tk = ImageTk.PhotoImage(img)
                self.preview_label.config(image=img_tk)
                self._preview_ref = img_tk
            except Exception:
                pass
        self.root.after(33, self._update_frame)

    def _show_no_camera(self) -> None:
        self.previewing = False
        self.preview_mode_label.config(text="✕ NO CAMERA", foreground=C["red"])
        self.preview_label.config(
            text="No Camera Connected",
            fg=C["subtext"],
            font=("Helvetica", 16),
            compound="center",
        )
        self._set_status("No camera detected — connect a camera and restart")

    def _resume_preview(self) -> None:
        if not CAMERA_AVAILABLE:
            return
        self.previewing = True
        self.preview_mode_label.config(text="● LIVE", foreground=C["green"])
        self._set_status("Live preview resumed — double-click to freeze")

    # ── Capture ───────────────────────────────────────────────────────────────

    def capture_image(self) -> None:
        self._set_status("Capturing…")
        threading.Thread(target=self._capture_worker, daemon=True).start()

    def _capture_worker(self) -> None:
        filepath = SAVE_DIR / f"image_{int(time.time())}.jpg"
        try:
            if self.cam:
                self.cam.capture_file(str(filepath))
            else:
                # Demo mode: solid grey plate stand-in
                Image.new("RGB", (640, 480), color=(50, 50, 70)).save(filepath)
            self.root.after(0, lambda: self._on_capture_done(filepath))
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"Capture error: {e}"))

    def _on_capture_done(self, filepath: Path) -> None:
        self.previewing = False
        self.preview_mode_label.config(text="◼ FROZEN", foreground=C["yellow"])
        self._set_status(f"Captured: {filepath.name}")
        self._show_image(filepath)
        self._add_thumbnail(filepath)
        self._select(filepath)

    # ── Gallery ───────────────────────────────────────────────────────────────

    def _add_thumbnail(self, filepath: Path) -> None:
        try:
            img = Image.open(filepath)
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img_tk = ImageTk.PhotoImage(img)

            cell = tk.Frame(self.gallery_inner, bg=C["surface"])
            cell.pack(side="left", padx=5, pady=6)

            border = tk.Frame(cell, bg=C["border"], padx=1, pady=1)
            border.pack()

            lbl = tk.Label(border, image=img_tk, bg=C["surface"], cursor="hand2")
            lbl.image = img_tk
            lbl.pack()

            tk.Label(cell, text=filepath.stem[-10:], bg=C["surface"],
                     fg=C["subtext"], font=("Helvetica", 7)).pack(pady=(2, 0))

            lbl.bind("<Button-1>", lambda e, p=filepath: self._select(p))
            self._thumb_refs.append(img_tk)

            self.gallery_canvas.update_idletasks()
            self.gallery_canvas.configure(
                scrollregion=self.gallery_canvas.bbox("all"))
            self.gallery_canvas.xview_moveto(1.0)
        except Exception as e:
            print("Thumbnail error:", e)

    def _select(self, filepath: Path) -> None:
        self.selected_capture = filepath
        self.previewing = False
        self.preview_mode_label.config(text="◼ FROZEN", foreground=C["yellow"])
        self._show_image(filepath)
        self.quantify_btn.config(state="normal")
        self._set_status(f"Selected: {filepath.name}  ·  Click Quantify to analyse")

    # ── Preview helpers ───────────────────────────────────────────────────────

    def _show_image(self, filepath: Path) -> None:
        try:
            img = Image.open(filepath).resize(
                (PREVIEW_W, PREVIEW_H), Image.BILINEAR)
            img_tk = ImageTk.PhotoImage(img)
            self.preview_label.config(image=img_tk)
            self._preview_ref = img_tk
        except Exception as e:
            print("Preview error:", e)

    # ── Quantify ──────────────────────────────────────────────────────────────

    def quantify_selected(self) -> None:
        if not self.selected_capture:
            return

        self.quantify_btn.config(state="disabled")
        self._set_status("Running colony analysis…")
        self._write_results("Analysing image, please wait…")
        threading.Thread(
            target=self._quantify_worker,
            args=(self.selected_capture,),
            daemon=True,
        ).start()

    def _quantify_worker(self, filepath: Path) -> None:
        out_path = RESULT_DIR / f"annotated_{filepath.name}"
        try:
            result = quantify_colonies(
                image_path=str(filepath),
                output_path=str(out_path),
            )
            # Layer 1 + 2 anomaly detection
            result = _detector.analyse(result)
            # Log features for ML training dataset
            _logger.log(result)
            self.root.after(0, lambda: self._on_quantify_done(result, out_path))
        except Exception as e:
            self.root.after(0, lambda: self._on_quantify_error(str(e)))

    def _on_quantify_done(self, result: dict, annotated: Path) -> None:
        count    = result["count"]
        s        = result.get("summary_stats", {})
        pc       = result.get("plate_circle", {})
        report   = result.get("anomaly_report", {})
        ml_on    = report.get("ml_active", False)
        colonies = result.get("contours", [])

        # ── Summary block ─────────────────────────────────────────────────────
        lines = [
            "╔══════════════════════════════╗",
            "║   COLONY ANALYSIS RESULTS    ║",
            "╚══════════════════════════════╝",
            "",
            f"  Colony Count       :  {count}",
            f"  Anomalies Flagged  :  {report.get('anomaly_count', 0)}"
            f"  ({report.get('anomaly_rate_pct', 0):.1f}%)",
            f"  High Confidence    :  {len(report.get('high_confidence', []))}",
            f"  Haemolysis Suspects:  {s.get('hemolysis_candidates', 0)}",
            "",
            "  Area (mm²)",
            f"    Mean             :  {s.get('mean_area_mm2', 0):>8.3f}",
            f"    Std Dev          :  {s.get('std_area_mm2',  0):>8.3f}",
            f"    Min              :  {s.get('min_area_mm2',  0):>8.3f}",
            f"    Max              :  {s.get('max_area_mm2',  0):>8.3f}",
            "",
            f"  Mean Circularity   :  {s.get('mean_circularity', 0):.3f}",
            f"  Plate Coverage     :  {s.get('coverage_pct', 0):.2f} %",
            f"  px / mm            :  {result.get('px_per_mm', 0):.2f}",
            f"  ML Layer           :  {'active' if ml_on else 'not yet trained'}",
            "",
            "  Plate geometry",
            f"    Outer radius     :  {pc.get('radius', 'N/A')} px",
            f"    Centre           :  ({pc.get('cx', '?')}, {pc.get('cy', '?')})",
        ]

        # ── Per-colony flag breakdown ─────────────────────────────────────────
        flagged = [(i + 1, c) for i, c in enumerate(colonies)
                   if c.get("anomaly_flags") or c.get("ml_anomaly")]
        normal_count = count - len(flagged)

        lines += ["", "─" * 36, "  COLONY FLAGS", "─" * 36]

        if flagged:
            lines.append(f"  {'#':<5} {'Area mm²':>8}  Flags")
            lines.append(f"  {'─'*5} {'─'*8}  {'─'*20}")
            for num, c in flagged[:50]:   # cap display at 50 rows
                all_flags = list(c.get("anomaly_flags", []))
                if c.get("ml_anomaly"):
                    all_flags.append("ml_anomaly")
                flag_str = " | ".join(all_flags)
                lines.append(f"  #{num:<4} {c['area_mm2']:>8.3f}  {flag_str}")
            if len(flagged) > 50:
                lines.append(f"  … {len(flagged) - 50} more flagged colonies")
        else:
            lines.append("  No flagged colonies")

        if normal_count > 0:
            lines.append(f"  [{normal_count} normal {'colony' if normal_count == 1 else 'colonies'} — no flags]")

        lines += ["", "  Annotated image saved → results/"]

        self._write_results("\n".join(lines))

        self.quantify_btn.config(state="normal")
        self._set_status(
            f"Analysis complete — {count} colonies, "
            f"{report.get('anomaly_count', 0)} anomalies detected")

        if annotated.exists():
            self._show_image(annotated)

    def _on_quantify_error(self, msg: str) -> None:
        self._write_results(f"Analysis failed:\n\n{msg}")
        self.quantify_btn.config(state="normal")
        self._set_status(f"Analysis error — see results panel")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _write_results(self, text: str) -> None:
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", text)
        self.results_text.config(state="disabled")

    def _set_status(self, msg: str) -> None:
        self.status_var.set(f"  {msg}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = PlateImagingApp(root)
    root.mainloop()

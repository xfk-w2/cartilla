"""
Generador y verificador de fotografías 35 x 45 mm para la Cartilla del Servicio
Militar Nacional (rostro de barbilla a nacimiento del cabello = 21 mm).

Flujo:
  1. Menú en terminal con las imágenes/PDF de la carpeta actual y la del script.
  2. Se toma UNA sola imagen (si es PDF, se rasteriza la primera página).
  3. Se miden barbilla (A) y nacimiento del cabello (B) con MediaPipe.
  4. Se escala y recorta a 413 x 531 px (35 x 45 mm a 300 DPI) con el rostro en 21 mm,
     sin máscaras ni alteración de fondo, ropa o sombras.
  5. La foto calibrada se duplica 5 veces (3 arriba + 2 abajo) en un PDF Carta al 100 %,
     junto con copia de inspección con cotas, reglas de 50 mm y reporte de cálculo.

Uso:
    python extractor.py                  # menú interactivo
    python extractor.py "ruta/foto.jpg"  # omite el menú

Dependencias:  pip install opencv-python mediapipe reportlab pillow pymupdf
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Silencia los mensajes internos de MediaPipe / TFLite
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@contextlib.contextmanager
def silenciar_stderr_c():
    """Redirige el stderr de bajo nivel (C/C++) a NUL mientras dure el bloque."""
    guardado = None
    try:
        nulo = os.open(os.devnull, os.O_RDWR)
        guardado = os.dup(2)
        os.dup2(nulo, 2)
        os.close(nulo)
    except OSError:
        guardado = None
    try:
        yield
    finally:
        if guardado is not None:
            os.dup2(guardado, 2)
            os.close(guardado)


import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp", ".pdf")

FOTO_ANCHO_MM = 35.0
FOTO_ALTO_MM = 45.0
ROSTRO_MM = 21.0               # barbilla -> nacimiento del cabello
CABELLO_DESDE_ARRIBA_MM = 9.0  # B queda a 9 mm del borde superior (A a 30 mm)
DPI = 300
NUM_FOTOS = 5

PX_POR_MM_NOMINAL = DPI / 25.4
LIENZO_W = round(FOTO_ANCHO_MM * PX_POR_MM_NOMINAL)   # 413 px
LIENZO_H = round(FOTO_ALTO_MM * PX_POR_MM_NOMINAL)    # 531 px
# Como el lienzo es de píxeles enteros, al colocarlo a 35 x 45 mm exactos la
# densidad real difiere de la nominal; la calibración usa la real.
PX_MM_X = LIENZO_W / FOTO_ANCHO_MM
PX_MM_Y = LIENZO_H / FOTO_ALTO_MM
DPI_EFECTIVO = PX_MM_Y * 25.4
TOLERANCIA_MM = 0.25

MODELO_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
MODELO_PATH = BASE_DIR / "modelos" / "face_landmarker.task"

SALIDA_PDF = BASE_DIR / "Cartilla_Militar_35x45mm_Listo.pdf"
CARPETA_FOTOS = BASE_DIR / "fotos_calibradas"

# Índices de la malla MediaPipe (478 puntos)
LM_FRENTE_SUP = 10
LM_FRENTE_MEDIA = 151
LM_GLABELA = 9
LM_SUBNASAL = 2
LM_MEJILLA_IZQ = 234
LM_MEJILLA_DER = 454
CONTORNO_MENTON = (152, 148, 176, 149, 150, 136, 377, 400, 378, 379, 365)


# ---------------------------------------------------------------------------
# Entrada: menú y lectura de archivos
# ---------------------------------------------------------------------------
def escanear_archivos() -> list[Path]:
    """Lista imágenes y PDF de la carpeta actual y de la del script (sin duplicados)."""
    vistos: set[Path] = set()
    encontrados: list[Path] = []
    for carpeta in (Path.cwd(), BASE_DIR):
        if not carpeta.is_dir():
            continue
        for f in sorted(carpeta.iterdir(), key=lambda p: p.name.lower()):
            real = f.resolve()
            if (f.is_file() and f.suffix.lower() in EXTENSIONES
                    and real not in vistos and real != SALIDA_PDF.resolve()):
                vistos.add(real)
                encontrados.append(real)
    return encontrados


def menu_seleccion() -> Path | None:
    """Muestra el menú numerado y devuelve la ruta elegida (None si se cancela)."""
    archivos = escanear_archivos()
    if not archivos:
        print("No se encontraron imágenes ni PDF en la carpeta actual ni en la del script.")
        return None

    print("\nArchivos encontrados en la carpeta:")
    for n, f in enumerate(archivos, 1):
        print(f"[{n}] {f.name}")
    print("[0] Cancelar / Salir")

    while True:
        try:
            texto = input("\nSeleccione el número del archivo: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not texto.isdigit():
            print("Entrada no válida: escriba solo un número.")
            continue
        opcion = int(texto)
        if opcion == 0:
            return None
        if 1 <= opcion <= len(archivos):
            return archivos[opcion - 1]
        print(f"Opción fuera de rango: elija entre 0 y {len(archivos)}.")


def rasterizar_pdf(ruta: Path) -> np.ndarray:
    """Convierte la primera página del PDF a imagen BGR."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Para leer PDF instale PyMuPDF:  pip install pymupdf") from exc
    with fitz.open(str(ruta)) as doc:
        pix = doc[0].get_pixmap(dpi=DPI, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def leer_imagen(ruta: Path) -> np.ndarray:
    """Lee imagen o PDF. Usa imdecode porque cv2.imread falla con rutas no ASCII en Windows."""
    if ruta.suffix.lower() == ".pdf":
        return rasterizar_pdf(ruta)
    img = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"El archivo no es una imagen válida: {ruta}")
    return img


def guardar_png(ruta: Path, img_bgr: np.ndarray, dpi: float) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(ruta, dpi=(dpi, dpi))


# ---------------------------------------------------------------------------
# Selección de la foto base dentro de la imagen
# ---------------------------------------------------------------------------
@dataclass
class Caja:
    x: int
    y: int
    w: int
    h: int


def detectar_fotos_en_hoja(img: np.ndarray) -> list[Caja]:
    """Busca recuadros de proporción 35:45 sobre papel blanco, ordenados por filas."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_tot, w_tot = gris.shape
    _, mascara = cv2.threshold(gris, 247, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cajas = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < w_tot * h_tot * 0.008:
            continue
        if abs(h / w - FOTO_ALTO_MM / FOTO_ANCHO_MM) > 0.12:
            continue
        if cv2.contourArea(cnt) / float(w * h) < 0.85:   # descarta bloques de texto
            continue
        m = 3                                            # recorta el filete del marco
        cajas.append(Caja(x + m, y + m, w - 2 * m, h - 2 * m))

    cajas.sort(key=lambda c: c.y)
    filas: list[list[Caja]] = []
    for c in cajas:
        if filas and abs(c.y - filas[-1][0].y) < c.h * 0.5:
            filas[-1].append(c)
        else:
            filas.append([c])
    return [c for f in filas for c in sorted(f, key=lambda c: c.x)]


def obtener_foto_base(img: np.ndarray) -> np.ndarray:
    """Devuelve la foto a procesar: la primera de una hoja con varias, o la imagen completa."""
    cajas = detectar_fotos_en_hoja(img)
    if len(cajas) >= 2:
        c = cajas[0]
        print(f"La imagen parece una hoja con {len(cajas)} fotos; se usa la primera detectada.")
        return img[c.y:c.y + c.h, c.x:c.x + c.w]
    return img


# ---------------------------------------------------------------------------
# Landmarks faciales y medición
# ---------------------------------------------------------------------------
class DetectorLandmarks:
    def __init__(self) -> None:
        try:
            with silenciar_stderr_c():
                import mediapipe as mp
                from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise RuntimeError("Falta MediaPipe:  pip install mediapipe") from exc
        if not MODELO_PATH.is_file():
            print(f"Descargando modelo de landmarks en {MODELO_PATH}")
            MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODELO_URL, MODELO_PATH)
        self._mp = mp
        with silenciar_stderr_c():
            opciones = vision.FaceLandmarkerOptions(
                # El modelo se pasa como buffer para evitar problemas con rutas no ASCII
                base_options=BaseOptions(model_asset_buffer=MODELO_PATH.read_bytes()),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )
            self._lm = vision.FaceLandmarker.create_from_options(opciones)

    def puntos(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """Array (478, 2) en píxeles, o None si no hay rostro."""
        h, w = img_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with silenciar_stderr_c():
            res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None
        return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], np.float64)

    def cerrar(self) -> None:
        self._lm.close()


@dataclass
class Medicion:
    A: np.ndarray         # barbilla (x, y) en px
    B: np.ndarray         # nacimiento del cabello (x, y) en px
    idx_A: int
    metodo_B: str
    centro_x: float       # centro horizontal del rostro
    puntos: np.ndarray

    @property
    def altura_px(self) -> float:
        return float(self.A[1] - self.B[1])

    @property
    def inclinacion_deg(self) -> float:
        d = self.B - self.A
        return math.degrees(math.atan2(d[0], -d[1]))


def _muestrear_eje(lab, origen, u, ts, semiancho, n=9) -> np.ndarray:
    """Color medio CIELAB en franjas perpendiculares al eje facial."""
    v = np.array([-u[1], u[0]])
    s = np.linspace(-semiancho, semiancho, n)
    mx = (origen[0] + ts[:, None] * u[0] + s[None, :] * v[0]).astype(np.float32)
    my = (origen[1] + ts[:, None] * u[1] + s[None, :] * v[1]).astype(np.float32)
    m = cv2.remap(lab, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return m.mean(axis=1)


def _color_piel(lab, centros, r) -> np.ndarray:
    h, w = lab.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    sel = np.zeros((h, w), bool)
    for cx, cy in centros:
        sel |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return np.median(lab[sel], axis=0)


def medir_puntos(img_bgr: np.ndarray, det: DetectorLandmarks) -> Medicion | None:
    pts = det.puntos(img_bgr)
    if pts is None:
        return None

    # A: punto más bajo del contorno del mentón
    idx_A = max(CONTORNO_MENTON, key=lambda i: pts[i, 1])
    A = pts[idx_A].copy()

    # B: la malla no incluye el trichion, así que se busca el borde piel -> cabello
    # sobre el eje facial, partiendo del landmark 10 y subiendo
    p10 = pts[LM_FRENTE_SUP]
    u = (p10 - A) / np.linalg.norm(p10 - A)
    ancho = np.linalg.norm(pts[LM_MEJILLA_DER] - pts[LM_MEJILLA_IZQ])
    lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    piel = _color_piel(lab, [pts[LM_FRENTE_MEDIA], pts[LM_GLABELA]], 0.03 * ancho)

    paso = 0.5
    t_ini = -float(np.linalg.norm(pts[LM_FRENTE_MEDIA] - p10))
    t_fin = 0.75 * float(np.linalg.norm(p10 - A))
    ts = np.arange(t_ini, t_fin, paso)
    perfil = _muestrear_eje(lab, p10, u, ts, 0.04 * ancho)
    dE = np.linalg.norm(perfil - piel, axis=1).astype(np.float32)
    dE = cv2.GaussianBlur(dE.reshape(-1, 1), (1, 5), 0).ravel()

    B, metodo = None, ""
    umbral, racha = 20.0, max(6, int(round(0.02 * np.linalg.norm(p10 - A) / paso)))
    for i in range(len(dE) - racha):
        if np.all(dE[i:i + racha] > umbral):
            base = float(np.median(dE[:max(1, i - 2)]))
            pelo = float(np.median(dE[i:i + 3 * racha]))
            mitad = (base + pelo) / 2.0          # el borde se ubica al 50 % de la transición
            j = i + racha
            while j > 1 and dE[j - 1] >= mitad:
                j -= 1
            f = (mitad - dE[j - 1]) / max(1e-6, dE[j] - dE[j - 1])
            t_b = ts[j - 1] + float(np.clip(f, 0, 1)) * paso
            B = p10 + u * t_b
            metodo = "borde piel-cabello (50 % de transición CIELAB)"
            break
    if B is None:
        # Sin contraste: se estima el tercio superior con glabela - subnasal
        B = pts[LM_GLABELA] + (pts[LM_GLABELA] - pts[LM_SUBNASAL])
        metodo = "ESTIMADO por tercios faciales (sin contraste piel/cabello)"

    centro_x = float((pts[LM_MEJILLA_IZQ, 0] + pts[LM_MEJILLA_DER, 0]) / 2.0)
    return Medicion(A, np.asarray(B, float), idx_A, metodo, centro_x, pts)


# ---------------------------------------------------------------------------
# Reencuadre 35 x 45 mm con rostro de 21 mm
# ---------------------------------------------------------------------------
def reencuadrar_foto(foto: np.ndarray, med: Medicion) -> tuple[np.ndarray, float]:
    """
    Escala la foto completa y extrae un rectángulo de 413 x 531 px centrado en el
    rostro, con B a 9 mm del borde superior. No se aplican máscaras ni se toca el
    fondo, la ropa o las sombras. Devuelve (foto calibrada, factor de escala).
    """
    s = ROSTRO_MM * PX_MM_Y / med.altura_px
    new_w = round(foto.shape[1] * s)
    new_h = round(foto.shape[0] * s)
    escalada = cv2.resize(foto, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    x1 = round(med.centro_x * s - LIENZO_W / 2.0)
    y1 = round(med.B[1] * s - CABELLO_DESDE_ARRIBA_MM * PX_MM_Y)
    x2, y2 = x1 + LIENZO_W, y1 + LIENZO_H

    izq, arr = max(0, -x1), max(0, -y1)
    der, aba = max(0, x2 - new_w), max(0, y2 - new_h)
    if izq or arr or der or aba:
        # Si el recorte se sale de la imagen, se rellena con el tono de las esquinas superiores
        esquinas = np.concatenate([
            escalada[: max(4, new_h // 20), : max(4, new_w // 10)],
            escalada[: max(4, new_h // 20), -max(4, new_w // 10):],
        ]).reshape(-1, 3)
        fondo = np.median(esquinas, axis=0).astype(np.uint8).tolist()
        escalada = cv2.copyMakeBorder(escalada, arr, aba, izq, der, cv2.BORDER_CONSTANT, value=fondo)
        x1, x2, y1, y2 = x1 + izq, x2 + izq, y1 + arr, y2 + arr

    return escalada[y1:y2, x1:x2].copy(), s


def guardar_debug(foto: np.ndarray, med: Medicion, ruta: Path) -> None:
    dbg = foto.copy()
    for x, y in med.puntos:
        cv2.circle(dbg, (int(x), int(y)), 1, (180, 180, 180), -1)
    for p, color, txt in ((med.A, (0, 170, 0), "A"), (med.B, (0, 0, 230), "B")):
        cv2.line(dbg, (0, int(round(p[1]))), (dbg.shape[1], int(round(p[1]))), color, 1)
        cv2.circle(dbg, (int(p[0]), int(p[1])), 4, color, -1)
        cv2.putText(dbg, txt, (int(p[0]) + 6, int(p[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    guardar_png(ruta, dbg, 72)


# ---------------------------------------------------------------------------
# PDF Carta a escala 100 %
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = letter

# Esquina superior izquierda (mm) de las 5 fotos: 3 arriba y 2 abajo
POSICIONES_FOTOS = [(25 + i * 47, 40) for i in range(3)] + [(25 + i * 47, 99) for i in range(2)]


def _P(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Convierte mm desde la esquina superior izquierda a puntos de ReportLab."""
    return x_mm * mm, PAGE_H - y_mm * mm


def _pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _marcas_corte(c, x_mm, y_mm) -> None:
    x, y = _P(x_mm, y_mm + FOTO_ALTO_MM)
    w, h = FOTO_ANCHO_MM * mm, FOTO_ALTO_MM * mm
    c.setStrokeColorRGB(0.80, 0.80, 0.80)
    c.setLineWidth(0.25)
    c.rect(x, y, w, h)
    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    sep, largo = 1 * mm, 3 * mm
    for cx, sx in ((x, -1), (x + w, 1)):
        for cy, sy in ((y, -1), (y + h, 1)):
            c.line(cx + sx * sep, cy, cx + sx * (sep + largo), cy)
            c.line(cx, cy + sy * sep, cx, cy + sy * (sep + largo))


def _regla(c, x_mm, y_mm, vertical: bool, largo_mm: int = 50) -> None:
    """Regla 1:1 con marcas cada 1, 5 y 10 mm."""
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 5.5)
    x0, y0 = _P(x_mm, y_mm)
    for i in range(largo_mm + 1):
        t = 4 if i % 10 == 0 else 2.6 if i % 5 == 0 else 1.5
        c.setLineWidth(0.35 if i % 5 == 0 else 0.2)
        if vertical:
            yy = y0 - i * mm
            c.line(x0, yy, x0 + t * mm, yy)
            if i % 10 == 0:
                c.drawString(x0 + 4.6 * mm, yy - 0.7 * mm, str(i))
        else:
            xx = x0 + i * mm
            c.line(xx, y0, xx, y0 + t * mm)
            if i % 10 == 0:
                c.drawCentredString(xx, y0 - 2.6 * mm, str(i))
    c.setLineWidth(0.4)
    if vertical:
        c.line(x0, y0, x0, y0 - largo_mm * mm)
        c.saveState()
        c.translate(x0 - 1.5 * mm, y0 - largo_mm / 2 * mm)
        c.rotate(90)
        c.drawCentredString(0, 0, f"REGLA DE CONTROL {largo_mm} mm (eje Y)")
        c.restoreState()
    else:
        c.line(x0, y0, x0 + largo_mm * mm, y0)
        c.drawString(x0 + (largo_mm + 3) * mm, y0 + 0.5 * mm,
                     f"REGLA DE CONTROL {largo_mm} mm (eje X) - compárela con una regla física")


def _flecha(c, x, y, ang, tam=1.3 * mm) -> None:
    """Punta de flecha rellena con el vértice en (x, y) apuntando hacia `ang`."""
    ca, sa = math.cos(ang), math.sin(ang)
    p = c.beginPath()
    p.moveTo(x, y)
    p.lineTo(x - tam * ca + 0.35 * tam * sa, y - tam * sa - 0.35 * tam * ca)
    p.lineTo(x - tam * ca - 0.35 * tam * sa, y - tam * sa + 0.35 * tam * ca)
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def _cota(c, p1, p2, punteada=False) -> None:
    (x1, y1), (x2, y2) = p1, p2
    if punteada:
        c.setDash(1.6, 1.2)
    c.line(x1, y1, x2, y2)
    c.setDash()
    ang = math.atan2(y2 - y1, x2 - x1)
    _flecha(c, x2, y2, ang)
    _flecha(c, x1, y1, ang + math.pi)


def _foto_inspeccion(c, foto, x0, y0, A_mm, B_mm, cara_mm, remedido_mm=None) -> None:
    """Copia de diagnóstico con cotas de 35 mm, 45 mm y del rostro."""
    w, h = FOTO_ANCHO_MM, FOTO_ALTO_MM
    xi, yi = _P(x0, y0 + h)
    c.drawImage(ImageReader(_pil(foto)), xi, yi, width=w * mm, height=h * mm)
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setLineWidth(0.3)
    c.rect(xi, yi, w * mm, h * mm)

    # Cota horizontal
    c.setLineWidth(0.25)
    for xx in (x0, x0 + w):
        c.line(*_P(xx, y0 - 0.8), *_P(xx, y0 - 7))
    _cota(c, _P(x0, y0 - 5), _P(x0 + w, y0 - 5))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawCentredString(*_P(x0 + w / 2, y0 - 6.2), f"[ {w:.1f} mm ]")

    # Cota vertical
    for yy in (y0, y0 + h):
        c.line(*_P(x0 - 0.8, yy), *_P(x0 - 8, yy))
    _cota(c, _P(x0 - 6, y0), _P(x0 - 6, y0 + h))
    c.saveState()
    c.translate(*_P(x0 - 7.2, y0 + h / 2))
    c.rotate(90)
    c.drawCentredString(0, 0, f"[ {h:.1f} mm ]")
    c.restoreState()

    # Líneas A y B con la cota del rostro
    xc = x0 + w + 6
    c.setStrokeColorRGB(0.85, 0.05, 0.05)
    c.setFillColorRGB(0.85, 0.05, 0.05)
    c.setLineWidth(0.3)
    c.setDash(1.2, 1.0)
    for yy in (B_mm, A_mm):
        c.line(*_P(x0, y0 + yy), *_P(xc + 2, y0 + yy))
    c.setDash()
    c.setLineWidth(0.35)
    _cota(c, _P(xc, y0 + B_mm), _P(xc, y0 + A_mm), punteada=True)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(*_P(xc + 2.5, y0 + (A_mm + B_mm) / 2 + 1.2), f"Cara: {cara_mm:.1f} mm")
    if remedido_mm is not None:
        c.setFont("Helvetica", 5.8)
        c.drawString(*_P(xc + 2.5, y0 + (A_mm + B_mm) / 2 + 4.2),
                     f"re-medición independiente: {remedido_mm:.2f} mm (tol. ±{TOLERANCIA_MM} mm)")
    c.setFont("Helvetica", 6)
    c.drawString(*_P(xc + 2.5, y0 + B_mm - 0.8), "B - nacimiento del cabello (trichion)")
    c.drawString(*_P(xc + 2.5, y0 + A_mm + 2.6), "A - barbilla / mentón")
    c.setFont("Helvetica", 5.5)
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.drawString(*_P(xc + 2.5, y0 + B_mm + 2.0), f"a {B_mm:.2f} mm del borde superior")
    c.drawString(*_P(xc + 2.5, y0 + A_mm + 4.8), f"a {A_mm:.2f} mm del borde superior")


def generar_pdf(foto: np.ndarray, diag: dict, salida: Path) -> None:
    c = canvas.Canvas(str(salida), pagesize=letter)
    c.setTitle("Cartilla SMN - Fotografías 35 x 45 mm con verificación")
    c.setViewerPreference("PrintScaling", "None")   # impresión al 100 %

    c.setFillColorRGB(0.15, 0.15, 0.15)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(*_P(20, 13), "Cartilla del Servicio Militar Nacional - Fotografías 35 x 45 mm")
    c.setFont("Helvetica", 7.5)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(*_P(20, 18), "Requisito: 35 x 45 mm, fondo blanco, barbilla al nacimiento del "
                 "cabello 21 mm.  Imprimir en papel Carta a escala 100 % / Tamaño real.")

    # Las 5 fotos son copias de la misma imagen calibrada
    imagen = ImageReader(_pil(foto))
    for x, y in POSICIONES_FOTOS[:NUM_FOTOS]:
        c.drawImage(imagen, *_P(x, y + FOTO_ALTO_MM),
                    width=FOTO_ANCHO_MM * mm, height=FOTO_ALTO_MM * mm)
        _marcas_corte(c, x, y)

    # Dos reglas: una impresora puede escalar X e Y de forma distinta
    _regla(c, 9, 40, vertical=True)
    _regla(c, 30, 270, vertical=False)

    # Copia de inspección
    c.setFillColorRGB(0.15, 0.15, 0.15)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(*_P(20, 154), "MODO INSPECCIÓN - copia de diagnóstico (NO recortar)")
    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.setLineWidth(0.3)
    c.line(*_P(20, 155.5), *_P(196, 155.5))
    _foto_inspeccion(c, foto, 32, 166, diag["A_dis"], diag["B_dis"],
                     diag["A_dis"] - diag["B_dis"], diag["cara_mm"])

    c.setFillColorRGB(0.15, 0.15, 0.15)
    c.setFont("Helvetica-Bold", 7.5)
    gx, gy = 128, 166
    c.drawString(*_P(gx, gy), "Comprobación con regla física:")
    c.setFont("Helvetica", 6.8)
    for k, t in enumerate([
        "1. Mida las reglas de control: 50 mm en X y 50 mm en Y.",
        "   Si no miden 50 mm, la impresión NO fue al 100 %.",
        "2. Mida cualquier foto: 35.0 mm de ancho x 45.0 mm de alto.",
        "3. Mida de la barbilla al nacimiento del cabello: 21.0 mm",
        "   (en esta copia, entre las dos líneas rojas punteadas).",
        "4. Recorte las 5 fotos por las marcas de las esquinas.",
    ]):
        c.drawString(*_P(gx, gy + 4.5 + k * 3.4), t)

    # Reporte de cálculo al pie
    bx, by, bw, bh = 20, 222, 176, 44
    c.setStrokeColorRGB(0.3, 0.3, 0.3)
    c.setFillColorRGB(0.97, 0.97, 0.97)
    c.setLineWidth(0.4)
    c.rect(*_P(bx, by + bh), bw * mm, bh * mm, stroke=1, fill=1)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(*_P(bx + 2.5, by + 4), "REPORTE DE CALIBRACIÓN Y CÁLCULO")
    c.setFont("Courier", 6.3)
    for k, linea in enumerate(diag["reporte"]):
        c.drawString(*_P(bx + 2.5, by + 8.2 + k * 2.95), linea)
    c.setFont("Helvetica-Bold", 8)
    if diag["cumple"]:
        c.setFillColorRGB(0.0, 0.5, 0.15)
    else:
        c.setFillColorRGB(0.8, 0.1, 0.1)
    c.drawRightString(*_P(bx + bw - 2.5, by + 4),
                      "RESULTADO: CUMPLE" if diag["cumple"] else "RESULTADO: REVISAR")

    c.showPage()
    c.save()


def verificar_pdf(salida: Path) -> None:
    """Relee el PDF y muestra los tamaños físicos reales de página e imágenes."""
    try:
        import fitz
    except ImportError:
        return
    with fitz.open(str(salida)) as d:
        p = d[0]
        print(f"    Página: {p.rect.width / 72 * 25.4:.2f} x {p.rect.height / 72 * 25.4:.2f} mm")
        for i, info in enumerate(p.get_image_info(xrefs=True), 1):
            x0, y0, x1, y1 = info["bbox"]
            print(f"    Imagen {i}: {(x1 - x0) / 72 * 25.4:.3f} x {(y1 - y0) / 72 * 25.4:.3f} mm"
                  f"  | {info['width']}x{info['height']} px")


# ---------------------------------------------------------------------------
# Reporte y programa principal
# ---------------------------------------------------------------------------
def construir_reporte(med0: Medicion, med1: Medicion | None, s: float) -> tuple[list[str], dict]:
    obj_px = ROSTRO_MM * PX_MM_Y
    if med1 is not None:
        A_mm, B_mm = med1.A[1] / PX_MM_Y, med1.B[1] / PX_MM_Y
        verif = f"{med1.altura_px:.2f} px / {PX_MM_Y:.4f} = {(A_mm - B_mm):.3f} mm"
    else:
        A_mm, B_mm = CABELLO_DESDE_ARRIBA_MM + ROSTRO_MM, CABELLO_DESDE_ARRIBA_MM
        verif = "no se pudo re-detectar el rostro (valores de diseño)"
    cara = A_mm - B_mm
    err = cara - ROSTRO_MM
    cumple = med1 is not None and abs(err) <= TOLERANCIA_MM and "ESTIMADO" not in med0.metodo_B
    r = [
        f"Resolución nominal ....... {DPI} DPI -> 1 mm = {DPI} / 25.4 = {PX_POR_MM_NOMINAL:.4f} px",
        f"Lienzo de la foto ........ round(35 x {PX_POR_MM_NOMINAL:.4f}) x round(45 x {PX_POR_MM_NOMINAL:.4f})"
        f" = {LIENZO_W} x {LIENZO_H} px",
        f"Densidad efectiva ........ {LIENZO_W}/35 = {PX_MM_X:.4f} px/mm (X)  {LIENZO_H}/45 = "
        f"{PX_MM_Y:.4f} px/mm (Y)  = {DPI_EFECTIVO:.2f} DPI",
        f"Rostro objetivo .......... 21.0 mm x {PX_MM_Y:.4f} = {obj_px:.2f} px",
        f"Medición en la foto base . A.y - B.y = {med0.altura_px:.2f} px "
        f"(inclinación {med0.inclinacion_deg:+.1f} grados)",
        f"Factor de escala ......... s = {obj_px:.2f} / {med0.altura_px:.2f} = {s:.4f} (uniforme, Lanczos-4)",
        f"Re-medición en salida .... {verif}   error {err:+.3f} mm (tol. +/-{TOLERANCIA_MM} mm)",
        f"Colocación en el PDF ..... 35*mm = {35 * mm:.3f} pt   45*mm = {45 * mm:.3f} pt   "
        f"21*mm = {21 * mm:.3f} pt",
        "Escala del PDF ........... PrintScaling=None (100 %), imágenes sin remuestreo",
        "Procesamiento ............ recorte directo, sin máscara alfa ni retoque de fondo/ropa",
        f"Punto A (barbilla) ....... landmark MediaPipe #{med0.idx_A} (punto más bajo del mentón)",
        f"Punto B (trichion) ....... {med0.metodo_B}",
    ]
    return r, {"A_mm": A_mm, "B_mm": B_mm, "cara_mm": cara, "cumple": cumple, "err": err,
               "B_dis": CABELLO_DESDE_ARRIBA_MM, "A_dis": CABELLO_DESDE_ARRIBA_MM + ROSTRO_MM}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", nargs="?", type=Path, help="Imagen o PDF (omite el menú)")
    ap.add_argument("-o", "--salida", type=Path, default=SALIDA_PDF, help="PDF de salida")
    args = ap.parse_args()

    ruta = args.entrada if args.entrada else menu_seleccion()
    if ruta is None:
        print("Operación cancelada.")
        return 0
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    print(f"\nArchivo seleccionado: {ruta.name}")

    foto_base = obtener_foto_base(leer_imagen(ruta))

    det = DetectorLandmarks()
    try:
        med = medir_puntos(foto_base, det)
        if med is None:
            print("ERROR: no se detectó ningún rostro en la imagen.", file=sys.stderr)
            return 1
        guardar_debug(foto_base, med, CARPETA_FOTOS / "debug_landmarks.png")
        final, s = reencuadrar_foto(foto_base, med)
        med_final = medir_puntos(final, det)    # segunda medición, independiente, sobre la salida
    finally:
        det.cerrar()

    reporte, d = construir_reporte(med, med_final, s)
    print(f"Foto base: {med.altura_px:.1f} px -> escala {s:.4f} -> rostro final "
          f"{d['cara_mm']:.3f} mm (error {d['err']:+.3f} mm) {'OK' if d['cumple'] else 'REVISAR'}")

    guardar_png(CARPETA_FOTOS / "foto_35x45mm.png", final, DPI_EFECTIVO)
    generar_pdf(final, {"reporte": reporte, **d}, args.salida)
    print(f"\nPDF generado: {args.salida}")
    verificar_pdf(args.salida)
    print(f"    Foto calibrada y diagnóstico: {CARPETA_FOTOS}")
    print("    Imprimir en papel Carta a escala 100 % ('Tamaño real', sin 'Ajustar a página').")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
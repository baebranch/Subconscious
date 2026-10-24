
"""
Image processing tools — resize, optimize, and convert images.
Supports raster formats via Pillow (PIL) and vector formats (SVG).

SVG support:
  - Resize / optimize: built-in via xml.etree.ElementTree + optional scour.
  - SVG → raster conversion: requires ``cairosvg`` (pip install cairosvg).
    On Windows, cairosvg also needs the GTK/Cairo runtime DLLs.
  - Raster → SVG conversion (auto-tracing): requires ``vtracer``
    (pip install vtracer). Works on all platforms with no system deps.

Resize policy (raster): downscale only — target dimensions must be smaller
than the original. Upscaling is rejected to prevent unintentional quality
loss. SVG resize is not subject to this restriction because SVGs are
resolution-independent.
"""

import io
import os
import pathlib
import logging
import platform
from PIL import Image
from typing import Optional
import xml.etree.ElementTree as ET
from pydantic_ai import RunContext

from . import EngineContext


logger = logging.getLogger("subconscious")

# ---------------------------------------------------------------------------
# Optional SVG dependencies
# ---------------------------------------------------------------------------

try:
  import scour.scour as _scour
  _SCOUR_AVAILABLE = True
except ImportError:
  _scour = None  # type: ignore[assignment]
  _SCOUR_AVAILABLE = False

try:
  import cairosvg as _cairosvg  #@IgnoreException
  _CAIROSVG_AVAILABLE = True
except (ImportError, OSError):
  _cairosvg = None  # type: ignore[assignment]
  _CAIROSVG_AVAILABLE = False

try:
  import vtracer as _vtracer
  _VTRACER_AVAILABLE = True
except ImportError:
  _vtracer = None  # type: ignore[assignment]
  _VTRACER_AVAILABLE = False

# PyMuPDF ("fitz") renders PDF pages to raster images. It is optional and
# lazily imported like the other heavy converters above.
#
# LICENSE NOTE: PyMuPDF is dual-licensed AGPL-3.0 / commercial (Artifex). AGPL
# is a much stronger copyleft than the LGPL packages already tracked in
# LGPL_COMPLIANCE.md — review/clear this with a commercial license (or swap to
# a non-AGPL renderer such as pdf2image+poppler) before shipping a build that
# bundles it.
try:
  import fitz as _fitz  # PyMuPDF
  _PYMUPDF_AVAILABLE = True
except ImportError:
  _fitz = None  # type: ignore[assignment]
  _PYMUPDF_AVAILABLE = False

if platform.system() == 'Windows':
  try:
    import nocairosvg as _nocairosvg
    _NOCAIROSVG_AVAILABLE = True
  except ImportError:
    _nocairosvg = None
    _NOCAIROSVG_AVAILABLE = False
else:
  _NOCAIROSVG_AVAILABLE = False


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

_SVG_EXTENSIONS = (".svg", ".svgz")
_SVG_NS = "http://www.w3.org/2000/svg"
_RASTER_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp", ".ico")
_ALL_IMAGE_EXTENSIONS = _RASTER_EXTENSIONS + _SVG_EXTENSIONS


def _is_svg(path: pathlib.Path) -> bool:
  """Return True if *path* is an SVG or SVGZ file."""
  return path.suffix.lower() in _SVG_EXTENSIONS


def _get_svg_dimensions(root: ET.Element) -> tuple[Optional[float], Optional[float]]:
  """
  Extract (width, height) from an SVG root element.
  Falls back to the viewBox if explicit attributes are absent.
  Returns (None, None) if dimensions cannot be determined.
  """
  def _parse_unit(val: str) -> Optional[float]:
    for unit in ("px", "pt", "mm", "cm", "in", "em", "rem", "%"):
      val = val.replace(unit, "")
    try:
      return float(val.strip())
    except ValueError:
      return None

  w_attr = root.get("width")
  h_attr = root.get("height")
  w = _parse_unit(w_attr) if w_attr else None
  h = _parse_unit(h_attr) if h_attr else None

  if w is None or h is None:
    vb = root.get("viewBox")
    if vb:
      parts = vb.replace(",", " ").split()
      if len(parts) == 4:
        try:
          w = float(parts[2])
          h = float(parts[3])
        except ValueError:
          pass

  return w, h


def _optimize_svg_bytes(svg_bytes: bytes) -> bytes:
  """
  Minify SVG content using scour if available; otherwise strip XML comments.
  """
  if _SCOUR_AVAILABLE:
    options = _scour.sanitizeOptions()  # type: ignore[union-attr]
    options.strip_comments = True
    options.remove_metadata = True
    options.shorten_ids = True
    options.indent_type = "none"
    options.newlines = False
    # Use scourString directly rather than start(): start() expects file-like
    # objects with a .name attribute (for its summary log) and closes them
    # afterward, which breaks in-memory BytesIO streams.
    out_string = _scour.scourString(svg_bytes.decode("utf-8"), options)  # type: ignore[union-attr]
    return out_string.encode("utf-8")

  # Lightweight fallback: strip XML comments
  import re
  text = svg_bytes.decode("utf-8", errors="replace")
  text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
  return text.encode("utf-8")


def _svg_to_raster_bytes(
  input_path: pathlib.Path,
  output_format: str,
  output_width: Optional[int] = None,
  output_height: Optional[int] = None,
  quality: int = 85,
) -> tuple[Optional[bytes], Optional[str]]:
  """
  Render an SVG to raster bytes using cairosvg or nocairosvg on Windows.

  Returns ``(bytes, None)`` on success or ``(None, error_message)`` on failure.
  """
  fmt = output_format.upper()
  kw: dict = {"url": str(input_path)}
  if output_width:
    kw["output_width"] = output_width
  if output_height:
    kw["output_height"] = output_height

  try:
    if platform.system() == 'Windows' and _NOCAIROSVG_AVAILABLE:
      if fmt == "PDF":
        data = _nocairosvg.svg2pdf(**kw)  # type: ignore[union-attr]
        return data, None

      png_data = _nocairosvg.svg2png(**kw)  # type: ignore[union-attr]

    elif _CAIROSVG_AVAILABLE:
      if fmt == "PDF":
        data = _cairosvg.svg2pdf(**kw)  # type: ignore[union-attr]
        return data, None

      png_data = _cairosvg.svg2png(**kw)  # type: ignore[union-attr]

    else:
      return None, (
        "SVG to raster conversion requires 'cairosvg' or 'nocairosvg' on Windows. "
        "Install cairosvg with: pip install cairosvg  "
        "(Windows also needs the GTK runtime: "
        "https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer) "
        "Or install nocairosvg for Windows."
      )

    if fmt == "PNG":
      return png_data, None

    with Image.open(io.BytesIO(png_data)) as img:
      if fmt in ("JPG", "JPEG"):
        if img.mode not in ("RGB", "L"):
          img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue(), None
      elif fmt == "WEBP":
        if img.mode in ("RGBA", "P"):
          img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=quality)
        return buf.getvalue(), None
      elif fmt == "BMP":
        buf = io.BytesIO()
        img.save(buf, "BMP")
        return buf.getvalue(), None
      else:
        return None, f"Cannot convert SVG to {fmt} via cairosvg or nocairosvg."

  except Exception as exc:
    lib = "nocairosvg" if platform.system() == 'Windows' and _NOCAIROSVG_AVAILABLE else "cairosvg"
    return None, f"{lib} render error: {exc}"


def _raster_to_svg(
  input_path: pathlib.Path,
  output_path: pathlib.Path,
  colormode: str = "color",
  filter_speckle: int = 4,
  color_precision: int = 6,
  layer_difference: int = 16,
  corner_threshold: int = 60,
  length_threshold: float = 4.0,
  splice_threshold: int = 45,
  path_precision: int = 8,
) -> Optional[str]:
  """
  Trace a raster image to SVG using vtracer.

  The input is first normalised to PNG (via Pillow) so that vtracer always
  receives a supported format regardless of the original file type.

  Args:
      input_path: Resolved path to the raster image.
      output_path: Resolved destination path for the .svg file.
      colormode: ``'color'`` (default) for full-colour tracing or
          ``'binary'`` for black-and-white tracing.
      filter_speckle: Discard patches smaller than this many pixels (default 4).
      color_precision: Number of significant bits for colour (1–8, default 6).
      layer_difference: Minimum brightness difference between layers (default 16).
      corner_threshold: Minimum angle (degrees) for a corner (default 60).
      length_threshold: Minimum path segment length (default 4.0).
      splice_threshold: Minimum angle (degrees) to splice a curve (default 45).
      path_precision: Decimal places in SVG path data (default 8).

  Returns:
      ``None`` on success, or an error string on failure.
  """
  if not _VTRACER_AVAILABLE:
    return (
      "Raster-to-SVG conversion requires 'vtracer'. "
      "Install it with: pip install vtracer"
    )

  try:
    # Normalise input to PNG bytes so vtracer gets a known-good format
    with Image.open(input_path) as img:
      # vtracer works best with RGBA
      if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA")
      buf = io.BytesIO()
      img.save(buf, "PNG")
      png_bytes = buf.getvalue()

    svg_str = _vtracer.convert_raw_image_to_svg(  # type: ignore[union-attr]
      png_bytes,
      img_format="png",
      colormode=colormode,
      hierarchical="stacked",
      mode="spline",
      filter_speckle=filter_speckle,
      color_precision=color_precision,
      layer_difference=layer_difference,
      corner_threshold=corner_threshold,
      length_threshold=length_threshold,
      max_iterations=10,
      splice_threshold=splice_threshold,
      path_precision=path_precision,
    )
    output_path.write_text(svg_str, encoding="utf-8")
    return None

  except Exception as exc:
    return f"vtracer error: {exc}"


async def optimize_image(
  ctx: RunContext[EngineContext],
  input_path: str,
  output_path: Optional[str] = None,
  quality: int = 85,
  max_size: int = 1024,
) -> str:
  """
  Optimize a single image file by capping its longest side at *max_size* pixels
  (aspect-ratio preserved) and saving with compression.

  For SVG files the image is minified/cleaned instead (the *max_size* parameter
  is ignored because SVGs are resolution-independent). Install ``scour`` for
  full SVG optimization; a lightweight comment-stripping fallback is used if
  scour is not available.

  The raster image is only ever downscaled — if the original is already smaller
  than *max_size* on both sides it is saved as-is without upscaling.

  Supports input/output formats: JPG, JPEG, PNG, BMP, TIFF, GIF, WebP, ICO, SVG.

  Args:
      input_path: Path to the input image file.
      output_path: Path for the optimized image. If None, overwrites input file.
      quality: Quality for JPEG/WebP (1-100, default 85). Ignored for SVG.
      max_size: Maximum pixel length of the longest side (default 1024). Ignored for SVG.

  Returns:
      Success message or error description.
  """
  try:
    input_p = pathlib.Path(input_path).expanduser().resolve()
    if not input_p.exists() or not input_p.is_file():
      return f"Error: Input file '{input_path}' does not exist or is not a file."

    if output_path is None:
      output_path = str(input_p)
    output_p = pathlib.Path(output_path).expanduser().resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # --- SVG path ---
    if _is_svg(input_p):
      optimized_bytes = _optimize_svg_bytes(input_p.read_bytes())
      output_p.write_bytes(optimized_bytes)
      engine = "scour" if _SCOUR_AVAILABLE else "lightweight fallback (install scour for full optimization)"
      return f"Successfully optimized SVG '{input_path}' using {engine} and saved to '{output_path}'."

    # --- Raster path ---
    with Image.open(input_p) as img:
      original_format = img.format or 'JPEG'
      orig_w, orig_h = img.size

      # Cap at max_size on the longest side, preserving aspect ratio.
      # Never upscale.
      scale = min(max_size / orig_w, max_size / orig_h, 1.0)
      new_w = int(orig_w * scale)
      new_h = int(orig_h * scale)
      optimized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

      if original_format.upper() in ('JPEG', 'JPG'):
        optimized.save(output_p, format='JPEG', quality=quality, optimize=True)
      elif original_format.upper() == 'PNG':
        optimized.save(output_p, format='PNG', optimize=True)
      elif original_format.upper() == 'ICO':
        if optimized.mode not in ('RGB', 'RGBA'):
          optimized = optimized.convert('RGBA')
        # ICO spec: max 256×256
        ico_size = min(new_w, new_h, 256)
        ico_img = optimized.resize((ico_size, ico_size), Image.Resampling.LANCZOS)
        ico_img.save(output_p, format='ICO')
      else:
        optimized.save(output_p, format=original_format)

    return f"Successfully optimized '{input_path}' and saved to '{output_path}'."

  except Exception as e:
    logger.error(f"Error optimizing image: {e}")
    return f"Error optimizing image: {e}"


async def convert_image(
  ctx: RunContext[EngineContext],
  input_path: str,
  output_format: str,
  output_path: Optional[str] = None,
  quality: int = 85,
) -> str:
  """
  Convert a single image file to the specified output format.

  SVG input → raster: requires ``cairosvg`` (pip install cairosvg). On Windows
  the GTK/Cairo runtime must also be installed.

  Raster → SVG (auto-tracing): requires ``vtracer`` (pip install vtracer).
  Works on all platforms with no system dependencies.

  Supported input formats: JPG, JPEG, PNG, BMP, TIFF, GIF, WebP, ICO, SVG
  Supported output formats: JPG, PNG, BMP, TIFF, GIF, WebP, ICO, PDF, SVG

  Args:
      input_path: Path to the input image file.
      output_format: Desired output format (case-insensitive: 'JPG', 'PNG', 'SVG', 'PDF', etc.).
      output_path: Path for the output file. If None, uses input path with new extension.
      quality: Quality for lossy formats like JPG/WebP (1-100, default 85). Ignored for lossless formats.

  Returns:
      Success message or error description.
  """
  try:
    input_p = pathlib.Path(input_path).expanduser().resolve()
    if not input_p.exists() or not input_p.is_file():
      return f"Error: Input file '{input_path}' does not exist or is not a file."

    # Normalize output format
    output_format = output_format.upper()
    supported_outputs = {'JPG', 'JPEG', 'PNG', 'BMP', 'TIFF', 'GIF', 'WEBP', 'ICO', 'PDF', 'SVG'}
    if output_format not in supported_outputs:
      return f"Error: Unsupported output format '{output_format}'. Supported: {', '.join(sorted(supported_outputs))}"

    # Determine output path
    if output_path is None:
      ext = '.jpg' if output_format in ('JPG', 'JPEG') else f'.{output_format.lower()}'
      output_path = str(input_p.with_suffix(ext))
    output_p = pathlib.Path(output_path).expanduser().resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # --- SVG input path ---
    if _is_svg(input_p):
      if output_format == 'SVG':
        import shutil
        shutil.copy2(input_p, output_p)
      else:
        svg_raster_formats = {'JPG', 'JPEG', 'PNG', 'WEBP', 'BMP', 'PDF'}
        if output_format not in svg_raster_formats:
          return (
            f"Error: Cannot convert SVG to {output_format}. "
            f"Supported SVG output formats: {', '.join(sorted(svg_raster_formats))}"
          )
        data, err = _svg_to_raster_bytes(input_p, output_format, quality=quality)
        if err:
          return f"Error: {err}"
        output_p.write_bytes(data)  # type: ignore[arg-type]
      return f"Successfully converted '{input_path}' to {output_format} at '{output_path}'."

    # --- Raster input path ---
    if output_format == 'SVG':
      err = _raster_to_svg(input_p, output_p)
      if err:
        return f"Error: {err}"
      return f"Successfully converted '{input_path}' to SVG at '{output_path}'."

    with Image.open(input_p) as img:
      # Handle format-specific requirements
      if output_format in ('JPG', 'JPEG'):
        # Convert to RGB for JPG
        if img.mode not in ('RGB', 'L'):
          img = img.convert('RGB')
        img.save(output_p, 'JPEG', quality=quality, optimize=True)
      elif output_format == 'PNG':
        img.save(output_p, 'PNG', optimize=True)
      elif output_format == 'WEBP':
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
          img = img.convert('RGB')
        img.save(output_p, 'WEBP', quality=quality, optimize=True)
      elif output_format == 'ICO':
        # Convert to RGBA if necessary for ICO
        if img.mode not in ('RGB', 'RGBA'):
          img = img.convert('RGBA')
        # Resize to standard ICO size if too large
        if img.size[0] > 256 or img.size[1] > 256:
          img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img.save(output_p, 'ICO')
      elif output_format == 'PDF':
        # Convert to RGB for PDF
        if img.mode in ('RGBA', 'P'):
          img = img.convert('RGB')
        img.save(output_p, 'PDF')
      else:
        # For BMP, TIFF, GIF - save as-is
        img.save(output_p, output_format)

    return f"Successfully converted '{input_path}' to {output_format} at '{output_path}'."

  except Exception as e:
    logger.error(f"Error converting image: {e}")
    return f"Error converting image: {e}"


async def batch_optimize_images(
  ctx: RunContext[EngineContext],
  src_directory: str = ".",
  dest_directory: str = "optimized",
  quality: int = 85,
  max_size: int = 1024,
) -> str:
  """
  Optimize all images in a source directory and save to destination directory.

  Args:
      src_directory: Source directory path.
      dest_directory: Destination directory path.
      quality: Quality for lossy formats (default 85).
      max_size: Maximum pixel length of the longest side (default 1024).

  Returns:
      Summary of operations.
  """
  try:
    src_p = pathlib.Path(src_directory).expanduser().resolve()
    dest_p = pathlib.Path(dest_directory).expanduser().resolve()
    dest_p.mkdir(parents=True, exist_ok=True)

    supported_formats = _ALL_IMAGE_EXTENSIONS
    processed = 0
    errors = []

    for filename in os.listdir(src_p):
      if filename.lower().endswith(supported_formats):
        input_path = src_p / filename
        output_path = dest_p / filename
        result = await optimize_image(ctx, str(input_path), str(output_path), quality, max_size)
        if "Successfully" in result:
          processed += 1
        else:
          errors.append(f"{filename}: {result}")

    result = f"Processed {processed} images."
    if errors:
      result += f" Errors: {', '.join(errors)}"
    return result

  except Exception as e:
    logger.error(f"Error in batch optimize: {e}")
    return f"Error in batch optimize: {e}"


async def batch_convert_image(
  ctx: RunContext[EngineContext],
  src_directory: str = ".",
  dest_directory: str = "converted",
  output_format: str = "JPG",
  quality: int = 85,
) -> str:
  """
  Convert all images in a source directory to the specified format and save to destination directory.

  Supported output formats: JPG, PNG, BMP, TIFF, GIF, WebP, ICO, PDF

  Args:
      src_directory: Source directory path.
      dest_directory: Destination directory path.
      output_format: Desired output format (case-insensitive).
      quality: Quality for lossy formats (default 85).

  Returns:
      Summary of operations.
  """
  try:
    src_p = pathlib.Path(src_directory).expanduser().resolve()
    dest_p = pathlib.Path(dest_directory).expanduser().resolve()
    dest_p.mkdir(parents=True, exist_ok=True)

    supported_formats = _ALL_IMAGE_EXTENSIONS
    processed = 0
    errors = []

    # Determine file extension for output format
    output_format = output_format.upper()
    if output_format in ('JPG', 'JPEG'):
      ext = '.jpg'
    elif output_format == 'PDF':
      ext = '.pdf'
    elif output_format == 'SVG':
      ext = '.svg'
    else:
      ext = f'.{output_format.lower()}'

    for filename in os.listdir(src_p):
      if filename.lower().endswith(supported_formats):
        input_path = src_p / filename
        new_filename = pathlib.Path(filename).with_suffix(ext)
        output_path = dest_p / new_filename
        try:
          result = await convert_image(ctx, str(input_path), output_format, str(output_path), quality)
          if "Successfully" in result:
            processed += 1
          else:
            errors.append(f"{filename}: {result}")
        except Exception as e:
          errors.append(f"{filename}: {e}")

    result = f"Converted {processed} images to {output_format}."
    if errors:
      result += f" Errors: {', '.join(errors)}"
    return result

  except Exception as e:
    logger.error(f"Error in batch convert to {output_format}: {e}")
    return f"Error in batch convert to {output_format}: {e}"


async def resize_image(
  ctx: RunContext[EngineContext],
  input_path: str,
  width: int,
  height: int,
  output_path: Optional[str] = None,
  maintain_aspect_ratio: bool = True,
) -> str:
  """
  Resize an image to the specified dimensions.

  For raster images, only downscaling is supported — the requested *width* and
  *height* must both be smaller than the original dimensions.

  For SVG files, any target size is accepted (upscaling is fine for vectors).
  The SVG's ``width``, ``height``, and ``viewBox`` attributes are updated in
  place while the underlying artwork scales perfectly.

  Pass either dimension as 0 to derive it automatically from the other while
  maintaining the aspect ratio.

  Args:
      input_path: Path to the input image file.
      width: Target width in pixels (0 = derive from height).
      height: Target height in pixels (0 = derive from width).
      output_path: Path for the resized image. If None, overwrites input file.
      maintain_aspect_ratio: When True (default) and both width and height are
          provided, the image is scaled so it fits within the given box without
          distortion. When False the image is stretched to the exact dimensions
          (for SVG this sets explicit width/height without altering viewBox).

  Returns:
      Success message with the final dimensions, or an error description.
  """
  try:
    input_p = pathlib.Path(input_path).expanduser().resolve()
    if not input_p.exists() or not input_p.is_file():
      return f"Error: Input file '{input_path}' does not exist or is not a file."

    if width < 0 or height < 0:
      return "Error: width and height must be non-negative integers."
    if width == 0 and height == 0:
      return "Error: at least one of width or height must be greater than zero."

    if output_path is None:
      output_path = str(input_p)
    output_p = pathlib.Path(output_path).expanduser().resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # --- SVG path ---
    if _is_svg(input_p):
      ET.register_namespace("", _SVG_NS)
      tree = ET.parse(input_p)
      root = tree.getroot()

      orig_w, orig_h = _get_svg_dimensions(root)

      # Derive missing dimension using original aspect ratio
      if width == 0 and orig_h and orig_w and height:
        width = int(orig_w * height / orig_h)
      elif height == 0 and orig_w and orig_h and width:
        height = int(orig_h * width / orig_w)
      elif maintain_aspect_ratio and orig_w and orig_h:
        scale = min(width / orig_w, height / orig_h)
        width = int(orig_w * scale)
        height = int(orig_h * scale)

      # Ensure viewBox is set so artwork scales with the new dimensions
      if root.get("viewBox") is None and orig_w and orig_h:
        root.set("viewBox", f"0 0 {orig_w} {orig_h}")

      root.set("width", str(width))
      root.set("height", str(height))

      tree.write(str(output_p), xml_declaration=True, encoding="unicode")
      return (
        f"Successfully resized SVG '{input_path}' to {width}\u00d7{height} "
        f"and saved to '{output_path}'."
      )

    # --- Raster path ---
    with Image.open(input_p) as img:
      orig_w, orig_h = img.size
      original_format = img.format or 'PNG'

      # Derive missing dimension
      if width == 0:
        scale = height / orig_h
        width = int(orig_w * scale)
      elif height == 0:
        scale = width / orig_w
        height = int(orig_h * scale)
      elif maintain_aspect_ratio:
        scale = min(width / orig_w, height / orig_h)
        width = int(orig_w * scale)
        height = int(orig_h * scale)

      # Enforce downscale-only
      if width > orig_w or height > orig_h:
        return (
          f"Error: Upscaling is not supported. "
          f"Requested {width}\u00d7{height} is larger than the original "
          f"{orig_w}\u00d7{orig_h}."
        )

      resized = img.resize((width, height), Image.Resampling.LANCZOS)
      resized.save(output_p, format=original_format)

    return (
      f"Successfully resized '{input_path}' to {width}\u00d7{height} "
      f"and saved to '{output_path}'."
    )

  except Exception as e:
    logger.error(f"Error resizing image: {e}")
    return f"Error resizing image: {e}"


async def batch_resize_images(
  ctx: RunContext[EngineContext],
  src_directory: str = ".",
  dest_directory: str = "resized",
  width: int = 0,
  height: int = 0,
  maintain_aspect_ratio: bool = True,
) -> str:
  """
  Resize all images in a source directory and save to destination directory.

  Only downscaling is supported. See resize_image() for dimension rules.

  Args:
      src_directory: Source directory path.
      dest_directory: Destination directory path.
      width: Target width in pixels (0 = derive from height).
      height: Target height in pixels (0 = derive from width).
      maintain_aspect_ratio: Preserve aspect ratio when both dimensions are given.

  Returns:
      Summary of operations.
  """
  try:
    src_p = pathlib.Path(src_directory).expanduser().resolve()
    dest_p = pathlib.Path(dest_directory).expanduser().resolve()
    dest_p.mkdir(parents=True, exist_ok=True)

    supported_formats = _ALL_IMAGE_EXTENSIONS
    processed = 0
    errors = []

    for filename in os.listdir(src_p):
      if filename.lower().endswith(supported_formats):
        input_path = src_p / filename
        output_path = dest_p / filename
        result = await resize_image(
          ctx, str(input_path), width, height,
          str(output_path), maintain_aspect_ratio
        )
        if "Successfully" in result:
          processed += 1
        else:
          errors.append(f"{filename}: {result}")

    result = f"Resized {processed} images."
    if errors:
      result += f" Errors: {', '.join(errors)}"
    return result

  except Exception as e:
    logger.error(f"Error in batch resize: {e}")
    return f"Error in batch resize: {e}"


async def images_to_pdf(
  ctx: RunContext[EngineContext],
  input_paths: list[str],
  output_path: str,
) -> str:
  """
  Combine one or more images into a single multi-page PDF, in the given order.

  Each input image becomes one PDF page sized to that image's pixel dimensions.
  SVG inputs are rendered to raster first (requires ``cairosvg``/``nocairosvg``
  on Windows) since PDF pages here are raster.

  Supported input formats: JPG, JPEG, PNG, BMP, TIFF, GIF, WebP, ICO, SVG.

  Args:
      input_paths: Ordered list of image file paths to combine, one per page.
      output_path: Path for the resulting PDF file.

  Returns:
      Success message or error description.
  """
  try:
    if not input_paths:
      return "Error: input_paths must contain at least one image."

    output_p = pathlib.Path(output_path).expanduser().resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)

    pages: list[Image.Image] = []
    try:
      for raw_path in input_paths:
        input_p = pathlib.Path(raw_path).expanduser().resolve()
        if not input_p.exists() or not input_p.is_file():
          return f"Error: Input file '{raw_path}' does not exist or is not a file."

        if _is_svg(input_p):
          data, err = _svg_to_raster_bytes(input_p, "PNG")
          if err:
            return f"Error: {err}"
          img = Image.open(io.BytesIO(data))  # type: ignore[arg-type]
        else:
          img = Image.open(input_p)

        if img.mode in ("RGBA", "P", "LA"):
          img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
          img = img.convert("RGB")
        pages.append(img)

      first, rest = pages[0], pages[1:]
      first.save(output_p, "PDF", save_all=True, append_images=rest)
    finally:
      for img in pages:
        img.close()

    return (
      f"Successfully combined {len(input_paths)} image(s) into PDF "
      f"at '{output_path}'."
    )

  except Exception as e:
    logger.error(f"Error converting images to PDF: {e}")
    return f"Error converting images to PDF: {e}"


async def pdf_to_images(
  ctx: RunContext[EngineContext],
  input_path: str,
  dest_directory: str,
  output_format: str = "PNG",
  dpi: int = 150,
) -> str:
  """
  Render each page of a PDF to a separate raster image file.

  Requires ``PyMuPDF`` (``pip install pymupdf``, module name ``fitz``).

  Note: PyMuPDF is dual-licensed AGPL-3.0 / commercial. Confirm licensing
  terms are acceptable for your distribution before relying on this tool in a
  shipped build.

  Output files are named ``<pdf-stem>_page_<n>.<ext>`` (1-indexed) inside
  *dest_directory*.

  Supported output formats: PNG, JPG, BMP, TIFF, WebP.

  Args:
      input_path: Path to the input PDF file.
      dest_directory: Directory to write the rendered page images into.
      output_format: Desired image format for each page (default 'PNG').
      dpi: Render resolution in dots per inch (default 150; higher = larger/sharper).

  Returns:
      Success message listing the number of pages rendered, or an error description.
  """
  if not _PYMUPDF_AVAILABLE:
    return (
      "PDF to image conversion requires 'PyMuPDF'. Install it with: "
      "pip install pymupdf  "
      "(Note: PyMuPDF is AGPL-3.0/commercial dual-licensed — review licensing "
      "terms before shipping.)"
    )

  try:
    input_p = pathlib.Path(input_path).expanduser().resolve()
    if not input_p.exists() or not input_p.is_file():
      return f"Error: Input file '{input_path}' does not exist or is not a file."

    output_format = output_format.upper()
    supported_outputs = {'JPG', 'JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP'}
    if output_format not in supported_outputs:
      return f"Error: Unsupported output format '{output_format}'. Supported: {', '.join(sorted(supported_outputs))}"
    ext = '.jpg' if output_format in ('JPG', 'JPEG') else f'.{output_format.lower()}'
    pil_format = 'JPEG' if output_format in ('JPG', 'JPEG') else output_format

    dest_p = pathlib.Path(dest_directory).expanduser().resolve()
    dest_p.mkdir(parents=True, exist_ok=True)

    stem = input_p.stem
    page_count = 0
    doc = _fitz.open(str(input_p))  # type: ignore[union-attr]
    try:
      for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        out_file = dest_p / f"{stem}_page_{i}{ext}"
        if output_format == "PNG":
          out_file.write_bytes(png_bytes)
        else:
          with Image.open(io.BytesIO(png_bytes)) as img:
            if pil_format == "JPEG" and img.mode in ("RGBA", "P"):
              img = img.convert("RGB")
            img.save(out_file, pil_format)
        page_count += 1
    finally:
      doc.close()

    return (
      f"Successfully rendered {page_count} page(s) from '{input_path}' "
      f"to '{dest_directory}' as {output_format}."
    )

  except Exception as e:
    logger.error(f"Error converting PDF to images: {e}")
    return f"Error converting PDF to images: {e}"


TOOLS = [
  optimize_image,
  convert_image,
  resize_image,
  batch_optimize_images,
  batch_convert_image,
  batch_resize_images,
  images_to_pdf,
  pdf_to_images,
]

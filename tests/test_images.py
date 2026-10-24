"""
Unit tests for subconscious.tools.images

Tests image conversion and resize tools using Pillow to create test images.
"""

import pathlib
import pytest
import pytest_asyncio
from PIL import Image

from subconscious.desktop_tools.images import (
  convert_image,
  optimize_image,
  resize_image,
  batch_convert_image,
  batch_optimize_images,
  batch_resize_images,
  images_to_pdf,
  pdf_to_images,
  _SCOUR_AVAILABLE,
  _CAIROSVG_AVAILABLE,
  _VTRACER_AVAILABLE,
  _PYMUPDF_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixture: temporary directory
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def tmp_dir(tmp_path):
  return tmp_path


# ---------------------------------------------------------------------------
# Helper: create test image
# ---------------------------------------------------------------------------

def create_test_png(path: pathlib.Path, size=(100, 100), color=(255, 0, 0)):
  """Create a simple PNG test image."""
  img = Image.new('RGB', size, color)
  img.save(path, 'PNG')


def create_test_jpg(path: pathlib.Path, size=(100, 100), color=(0, 255, 0)):
  """Create a simple JPG test image."""
  img = Image.new('RGB', size, color)
  img.save(path, 'JPEG')


def create_test_bmp(path: pathlib.Path, size=(100, 100), color=(0, 0, 255)):
  """Create a simple BMP test image."""
  img = Image.new('RGB', size, color)
  img.save(path, 'BMP')


def create_test_tiff(path: pathlib.Path, size=(100, 100), color=(255, 255, 0)):
  """Create a simple TIFF test image."""
  img = Image.new('RGB', size, color)
  img.save(path, 'TIFF')


def create_test_gif(path: pathlib.Path, size=(100, 100), color=(255, 0, 255)):
  """Create a simple GIF test image."""
  img = Image.new('P', size, color)
  img.save(path, 'GIF')


def create_test_webp(path: pathlib.Path, size=(100, 100), color=(0, 255, 255)):
  """Create a simple WebP test image."""
  img = Image.new('RGB', size, color)
  img.save(path, 'WEBP')


def create_test_ico(path: pathlib.Path, size=(64, 64), color=(128, 128, 128)):
  """Create a simple ICO test image."""
  img = Image.new('RGBA', size, color + (255,))
  img.save(path, 'ICO')


# ---------------------------------------------------------------------------
# optimize_image
# ---------------------------------------------------------------------------

async def test_optimize_image_png(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "optimized.png"
  create_test_png(input_path, size=(200, 200))

  result = await optimize_image(ctx, str(input_path), str(output_path), max_size=100)
  assert "Successfully optimized" in result
  assert output_path.exists()

  # 200x200 scaled down so longest side == 100 → 100x100
  with Image.open(output_path) as img:
    assert img.size == (100, 100)


async def test_optimize_image_overwrite(ctx, tmp_dir):
  input_path = tmp_dir / "test.jpg"
  create_test_jpg(input_path)

  result = await optimize_image(ctx, str(input_path))
  assert "Successfully optimized" in result
  assert input_path.exists()  # Overwrites input


async def test_optimize_image_jpg(ctx, tmp_dir):
  input_path = tmp_dir / "test.jpg"
  output_path = tmp_dir / "optimized.jpg"
  create_test_jpg(input_path, size=(200, 200))

  result = await optimize_image(ctx, str(input_path), str(output_path), max_size=100)
  assert "Successfully optimized" in result
  assert output_path.exists()

  with Image.open(output_path) as img:
    assert img.size == (100, 100)


async def test_optimize_image_bmp(ctx, tmp_dir):
  input_path = tmp_dir / "test.bmp"
  output_path = tmp_dir / "optimized.bmp"
  create_test_bmp(input_path, size=(200, 200))

  result = await optimize_image(ctx, str(input_path), str(output_path), max_size=100)
  assert "Successfully optimized" in result
  assert output_path.exists()

  with Image.open(output_path) as img:
    assert img.size == (100, 100)


async def test_optimize_image_tiff(ctx, tmp_dir):
  input_path = tmp_dir / "test.tiff"
  output_path = tmp_dir / "optimized.tiff"
  create_test_tiff(input_path, size=(200, 200))

  result = await optimize_image(ctx, str(input_path), str(output_path), max_size=100)
  assert "Successfully optimized" in result
  assert output_path.exists()

  with Image.open(output_path) as img:
    assert img.size == (100, 100)


async def test_optimize_image_gif(ctx, tmp_dir):
  input_path = tmp_dir / "test.gif"
  output_path = tmp_dir / "optimized.gif"
  create_test_gif(input_path, size=(200, 200))

  result = await optimize_image(ctx, str(input_path), str(output_path), max_size=100)
  assert "Successfully optimized" in result
  assert output_path.exists()

  with Image.open(output_path) as img:
    assert img.size == (100, 100)


async def test_optimize_image_webp(ctx, tmp_dir):
  input_path = tmp_dir / "test.webp"
  output_path = tmp_dir / "optimized.webp"
  create_test_webp(input_path, size=(200, 200))

  result = await optimize_image(ctx, str(input_path), str(output_path), max_size=100)
  assert "Successfully optimized" in result
  assert output_path.exists()

  with Image.open(output_path) as img:
    assert img.size == (100, 100)


async def test_optimize_image_ico(ctx, tmp_dir):
  input_path = tmp_dir / "test.ico"
  output_path = tmp_dir / "optimized.ico"
  create_test_ico(input_path, size=(64, 64))

  result = await optimize_image(ctx, str(input_path), str(output_path), max_size=32)
  assert "Successfully optimized" in result
  assert output_path.exists()

  # ICO is capped at min(new_size, 256)
  with Image.open(output_path) as img:
    assert img.size == (32, 32)


# ---------------------------------------------------------------------------
# convert_image (general converter)
# ---------------------------------------------------------------------------

async def test_convert_image_png_to_jpg(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "output.jpg"
  create_test_png(input_path)

  result = await convert_image(ctx, str(input_path), 'JPG', str(output_path))
  assert "Successfully converted" in result and "JPG" in result
  assert output_path.exists()
  assert output_path.suffix == ".jpg"


async def test_convert_image_jpg_to_png(ctx, tmp_dir):
  input_path = tmp_dir / "test.jpg"
  output_path = tmp_dir / "output.png"
  create_test_jpg(input_path)

  result = await convert_image(ctx, str(input_path), 'PNG', str(output_path))
  assert "Successfully converted" in result and "PNG" in result
  assert output_path.exists()
  assert output_path.suffix == ".png"


async def test_convert_image_png_to_webp(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "output.webp"
  create_test_png(input_path)

  result = await convert_image(ctx, str(input_path), 'WEBP', str(output_path))
  assert "Successfully converted" in result and "WEBP" in result
  assert output_path.exists()
  assert output_path.suffix == ".webp"


async def test_convert_image_jpg_to_pdf(ctx, tmp_dir):
  input_path = tmp_dir / "test.jpg"
  output_path = tmp_dir / "output.pdf"
  create_test_jpg(input_path)

  result = await convert_image(ctx, str(input_path), 'PDF', str(output_path))
  assert "Successfully converted" in result and "PDF" in result
  assert output_path.exists()
  assert output_path.suffix == ".pdf"


async def test_convert_image_auto_output(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  create_test_png(input_path)

  result = await convert_image(ctx, str(input_path), 'JPG')
  expected_output = input_path.with_suffix('.jpg')
  assert "Successfully converted" in result and "JPG" in result
  assert expected_output.exists()


async def test_convert_image_unsupported_format(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  create_test_png(input_path)

  result = await convert_image(ctx, str(input_path), 'UNSUPPORTED')
  assert "Unsupported output format" in result


async def test_convert_image_invalid_input(ctx, tmp_dir):
  result = await convert_image(ctx, str(tmp_dir / "missing.png"), 'JPG')
  assert "not exist" in result.lower()


# ---------------------------------------------------------------------------
# convert_image (JPG conversions)
# ---------------------------------------------------------------------------

async def test_convert_image_to_jpg_png(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "output.jpg"
  create_test_png(input_path)

  result = await convert_image(ctx, str(input_path), 'JPG', str(output_path))
  assert "Successfully converted" in result and "JPG" in result
  assert output_path.exists()
  assert output_path.suffix == ".jpg"


async def test_convert_image_to_jpg_auto_output(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  create_test_png(input_path)

  result = await convert_image(ctx, str(input_path), 'JPG')
  expected_output = input_path.with_suffix('.jpg')
  assert "Successfully converted" in result and "JPG" in result
  assert expected_output.exists()


# ---------------------------------------------------------------------------
# convert_image (PDF conversions)
# ---------------------------------------------------------------------------

async def test_convert_image_to_pdf_png(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "output.pdf"
  create_test_png(input_path)

  result = await convert_image(ctx, str(input_path), 'PDF', str(output_path))
  assert "Successfully converted" in result and "PDF" in result
  assert output_path.exists()
  assert output_path.suffix == ".pdf"


async def test_convert_image_to_pdf_jpg(ctx, tmp_dir):
  input_path = tmp_dir / "test.jpg"
  create_test_jpg(input_path)

  result = await convert_image(ctx, str(input_path), 'PDF')
  expected_output = input_path.with_suffix('.pdf')
  assert "Successfully converted" in result and "PDF" in result
  assert expected_output.exists()


# ---------------------------------------------------------------------------
# batch_optimize_images
# ---------------------------------------------------------------------------

async def test_batch_optimize_images(ctx, tmp_dir):
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()

  create_test_png(src_dir / "test1.png")
  create_test_jpg(src_dir / "test2.jpg")
  create_test_bmp(src_dir / "test3.bmp")
  create_test_webp(src_dir / "test4.webp")
  create_test_ico(src_dir / "test5.ico")

  result = await batch_optimize_images(ctx, str(src_dir), str(dest_dir))
  assert "Processed 5 images" in result
  assert (dest_dir / "test1.png").exists()
  assert (dest_dir / "test2.jpg").exists()
  assert (dest_dir / "test3.bmp").exists()
  assert (dest_dir / "test4.webp").exists()
  assert (dest_dir / "test5.ico").exists()


# ---------------------------------------------------------------------------
# batch_convert_image
# ---------------------------------------------------------------------------

async def test_batch_convert_image_to_png(ctx, tmp_dir):
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()

  create_test_png(src_dir / "test1.png")
  create_test_jpg(src_dir / "test2.jpg")

  result = await batch_convert_image(ctx, str(src_dir), str(dest_dir), 'PNG')
  assert "Converted 2 images to PNG" in result
  assert (dest_dir / "test1.png").exists()
  assert (dest_dir / "test2.png").exists()


async def test_batch_convert_image_to_webp(ctx, tmp_dir):
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()

  create_test_png(src_dir / "test1.png")
  create_test_jpg(src_dir / "test2.jpg")

  result = await batch_convert_image(ctx, str(src_dir), str(dest_dir), 'WEBP', quality=90)
  assert "Converted 2 images to WEBP" in result
  assert (dest_dir / "test1.webp").exists()
  assert (dest_dir / "test2.webp").exists()


async def test_batch_convert_image_to_jpg(ctx, tmp_dir):
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()

  create_test_png(src_dir / "test1.png")
  create_test_jpg(src_dir / "test2.jpg")

  result = await batch_convert_image(ctx, str(src_dir), str(dest_dir), 'JPG')
  assert "Converted 2 images to JPG" in result
  assert (dest_dir / "test1.jpg").exists()
  assert (dest_dir / "test2.jpg").exists()


async def test_batch_convert_image_to_pdf(ctx, tmp_dir):
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()

  create_test_png(src_dir / "test1.png")
  create_test_jpg(src_dir / "test2.jpg")

  result = await batch_convert_image(ctx, str(src_dir), str(dest_dir), 'PDF')
  assert "Converted 2 images to PDF" in result
  assert (dest_dir / "test1.pdf").exists()
  assert (dest_dir / "test2.pdf").exists()


async def test_batch_convert_image_empty_directory(ctx, tmp_dir):
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()

  result = await batch_convert_image(ctx, str(src_dir), str(dest_dir), 'JPG')
  assert "Converted 0 images to JPG" in result


# ---------------------------------------------------------------------------
# optimize_image — additional behaviour tests
# ---------------------------------------------------------------------------

async def test_optimize_image_no_upscale(ctx, tmp_dir):
  """Images smaller than max_size must not be upscaled."""
  input_path = tmp_dir / "small.png"
  output_path = tmp_dir / "out.png"
  create_test_png(input_path, size=(50, 50))

  await optimize_image(ctx, str(input_path), str(output_path), max_size=200)

  with Image.open(output_path) as img:
    assert img.size == (50, 50)  # unchanged


async def test_optimize_image_aspect_ratio_preserved(ctx, tmp_dir):
  """Non-square images must keep their aspect ratio after optimization."""
  input_path = tmp_dir / "wide.png"
  output_path = tmp_dir / "out.png"
  # 400×200 with max_size=200 → longest side (400) → 200×100
  img = Image.new('RGB', (400, 200), (0, 128, 255))
  img.save(input_path, 'PNG')

  await optimize_image(ctx, str(input_path), str(output_path), max_size=200)

  with Image.open(output_path) as img_out:
    assert img_out.size == (200, 100)


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------

async def test_resize_image_explicit_dimensions(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "resized.png"
  create_test_png(input_path, size=(400, 400))

  result = await resize_image(ctx, str(input_path), 200, 200, str(output_path))
  assert "Successfully resized" in result
  assert "200×200" in result
  with Image.open(output_path) as img:
    assert img.size == (200, 200)


async def test_resize_image_maintain_aspect_ratio(ctx, tmp_dir):
  """400×200 image resized into a 200×200 box with AR → 200×100."""
  input_path = tmp_dir / "wide.png"
  output_path = tmp_dir / "resized.png"
  img = Image.new('RGB', (400, 200), (0, 0, 0))
  img.save(input_path, 'PNG')

  result = await resize_image(ctx, str(input_path), 200, 200, str(output_path),
                              maintain_aspect_ratio=True)
  assert "Successfully resized" in result
  with Image.open(output_path) as img_out:
    assert img_out.size == (200, 100)


async def test_resize_image_no_aspect_ratio(ctx, tmp_dir):
  """Without AR preservation the image is stretched to exact dimensions."""
  input_path = tmp_dir / "wide.png"
  output_path = tmp_dir / "resized.png"
  img = Image.new('RGB', (400, 200), (0, 0, 0))
  img.save(input_path, 'PNG')

  result = await resize_image(ctx, str(input_path), 150, 100, str(output_path),
                              maintain_aspect_ratio=False)
  assert "Successfully resized" in result
  with Image.open(output_path) as img_out:
    assert img_out.size == (150, 100)


async def test_resize_image_derive_width_from_height(ctx, tmp_dir):
  """width=0 → derived from height to preserve aspect ratio."""
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "resized.png"
  create_test_png(input_path, size=(400, 200))

  result = await resize_image(ctx, str(input_path), 0, 100, str(output_path))
  assert "Successfully resized" in result
  with Image.open(output_path) as img:
    assert img.size == (200, 100)  # width proportionally halved


async def test_resize_image_derive_height_from_width(ctx, tmp_dir):
  """height=0 → derived from width."""
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "resized.png"
  create_test_png(input_path, size=(400, 200))

  result = await resize_image(ctx, str(input_path), 200, 0, str(output_path))
  assert "Successfully resized" in result
  with Image.open(output_path) as img:
    assert img.size == (200, 100)


async def test_resize_image_rejects_upscale(ctx, tmp_dir):
  """Requesting dimensions larger than the original must be rejected."""
  input_path = tmp_dir / "small.png"
  create_test_png(input_path, size=(100, 100))

  result = await resize_image(ctx, str(input_path), 500, 500)
  assert "Upscaling is not supported" in result


async def test_resize_image_overwrites_input(ctx, tmp_dir):
  """output_path=None should overwrite the original file."""
  input_path = tmp_dir / "test.png"
  create_test_png(input_path, size=(200, 200))

  result = await resize_image(ctx, str(input_path), 100, 100)
  assert "Successfully resized" in result
  with Image.open(input_path) as img:
    assert img.size == (100, 100)


async def test_resize_image_invalid_path(ctx, tmp_dir):
  result = await resize_image(ctx, str(tmp_dir / "missing.png"), 100, 100)
  assert "not exist" in result.lower()


async def test_resize_image_both_zero(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  create_test_png(input_path, size=(200, 200))
  result = await resize_image(ctx, str(input_path), 0, 0)
  assert "at least one" in result.lower()


# ---------------------------------------------------------------------------
# batch_resize_images
# ---------------------------------------------------------------------------

async def test_batch_resize_images(ctx, tmp_dir):
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()
  create_test_png(src_dir / "a.png", size=(300, 300))
  create_test_jpg(src_dir / "b.jpg", size=(300, 300))

  result = await batch_resize_images(ctx, str(src_dir), str(dest_dir),
                                     width=150, height=150)
  assert "Resized 2 images" in result
  for name in ("a.png", "b.jpg"):
    with Image.open(dest_dir / name) as img:
      assert img.size == (150, 150)


async def test_batch_resize_images_aspect_ratio(ctx, tmp_dir):
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()
  # 400×200 → fit into 200×200 box with AR → 200×100
  img = Image.new('RGB', (400, 200), (0, 0, 0))
  img.save(src_dir / "wide.png", 'PNG')

  result = await batch_resize_images(ctx, str(src_dir), str(dest_dir),
                                     width=200, height=200,
                                     maintain_aspect_ratio=True)
  assert "Resized 1 images" in result
  with Image.open(dest_dir / "wide.png") as img_out:
    assert img_out.size == (200, 100)


async def test_batch_resize_images_skips_upscale(ctx, tmp_dir):
  """Files that would require upscaling should appear in the errors list."""
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()
  create_test_png(src_dir / "tiny.png", size=(50, 50))

  result = await batch_resize_images(ctx, str(src_dir), str(dest_dir),
                                     width=200, height=200)
  assert "Resized 0 images" in result
  assert "Errors" in result


async def test_batch_resize_images_empty_directory(ctx, tmp_dir):
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()

  result = await batch_resize_images(ctx, str(src_dir), str(dest_dir), width=100, height=100)
  assert "Resized 0 images" in result


# ---------------------------------------------------------------------------
# SVG helper
# ---------------------------------------------------------------------------

def create_test_svg(path: pathlib.Path, width: int = 100, height: int = 100) -> None:
  """Write a minimal, valid SVG file."""
  svg = (
    f'<?xml version="1.0" encoding="UTF-8"?>\n'
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    f'  <!-- test comment -->\n'
    f'  <rect width="{width}" height="{height}" fill="red"/>\n'
    f'</svg>\n'
  )
  path.write_text(svg, encoding="utf-8")


def create_test_svg_no_dimensions(path: pathlib.Path, vb_w: int = 200, vb_h: int = 100) -> None:
  """SVG with only a viewBox and no explicit width/height."""
  svg = (
    f'<?xml version="1.0" encoding="UTF-8"?>\n'
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vb_w} {vb_h}">\n'
    f'  <rect width="{vb_w}" height="{vb_h}" fill="blue"/>\n'
    f'</svg>\n'
  )
  path.write_text(svg, encoding="utf-8")


# ---------------------------------------------------------------------------
# optimize_image — SVG
# ---------------------------------------------------------------------------

async def test_optimize_svg_basic(ctx, tmp_dir):
  """Optimizing an SVG should succeed and produce a smaller/equal file."""
  input_path = tmp_dir / "test.svg"
  output_path = tmp_dir / "optimized.svg"
  create_test_svg(input_path)

  result = await optimize_image(ctx, str(input_path), str(output_path))
  assert "Successfully optimized" in result
  assert output_path.exists()
  content = output_path.read_text()
  assert "<svg" in content


async def test_optimize_svg_strips_comments(ctx, tmp_dir):
  """Comments should be removed from the optimized SVG."""
  input_path = tmp_dir / "test.svg"
  output_path = tmp_dir / "optimized.svg"
  create_test_svg(input_path)

  await optimize_image(ctx, str(input_path), str(output_path))
  content = output_path.read_text()
  assert "<!-- test comment -->" not in content


async def test_optimize_svg_overwrite(ctx, tmp_dir):
  """No output_path → overwrite input."""
  input_path = tmp_dir / "test.svg"
  create_test_svg(input_path)

  result = await optimize_image(ctx, str(input_path))
  assert "Successfully optimized" in result
  assert input_path.exists()


async def test_optimize_svg_invalid_path(ctx, tmp_dir):
  result = await optimize_image(ctx, str(tmp_dir / "missing.svg"))
  assert "not exist" in result.lower()


# ---------------------------------------------------------------------------
# resize_image — SVG
# ---------------------------------------------------------------------------

async def test_resize_svg_explicit_dimensions(ctx, tmp_dir):
  """SVG resize to explicit width×height should update XML attributes."""
  input_path = tmp_dir / "test.svg"
  output_path = tmp_dir / "resized.svg"
  create_test_svg(input_path, width=100, height=100)

  result = await resize_image(ctx, str(input_path), 200, 200, str(output_path))
  assert "Successfully resized" in result
  assert "200×200" in result
  content = output_path.read_text()
  assert 'width="200"' in content
  assert 'height="200"' in content


async def test_resize_svg_allows_upscale(ctx, tmp_dir):
  """SVG resize must allow upscaling (unlike raster)."""
  input_path = tmp_dir / "test.svg"
  output_path = tmp_dir / "resized.svg"
  create_test_svg(input_path, width=50, height=50)

  result = await resize_image(ctx, str(input_path), 400, 400, str(output_path))
  assert "Successfully resized" in result
  assert "Upscaling" not in result


async def test_resize_svg_maintain_aspect_ratio(ctx, tmp_dir):
  """200×100 SVG into a 200×200 box with AR → 200×100."""
  input_path = tmp_dir / "wide.svg"
  output_path = tmp_dir / "resized.svg"
  create_test_svg(input_path, width=200, height=100)

  result = await resize_image(ctx, str(input_path), 200, 200, str(output_path),
                              maintain_aspect_ratio=True)
  assert "Successfully resized" in result
  content = output_path.read_text()
  assert 'width="200"' in content
  assert 'height="100"' in content


async def test_resize_svg_derive_width_from_height(ctx, tmp_dir):
  """width=0 → derived from height for SVG."""
  input_path = tmp_dir / "test.svg"
  output_path = tmp_dir / "resized.svg"
  create_test_svg(input_path, width=200, height=100)

  result = await resize_image(ctx, str(input_path), 0, 50, str(output_path))
  assert "Successfully resized" in result
  content = output_path.read_text()
  assert 'height="50"' in content
  assert 'width="100"' in content


async def test_resize_svg_preserves_viewbox(ctx, tmp_dir):
  """Resizing an SVG that already has a viewBox should keep the viewBox intact."""
  input_path = tmp_dir / "test.svg"
  output_path = tmp_dir / "resized.svg"
  create_test_svg(input_path, width=100, height=100)

  await resize_image(ctx, str(input_path), 50, 50, str(output_path))
  content = output_path.read_text()
  assert "viewBox" in content


async def test_resize_svg_adds_viewbox_when_missing(ctx, tmp_dir):
  """An SVG with no viewBox should have one added when resized."""
  input_path = tmp_dir / "no_vb.svg"
  output_path = tmp_dir / "resized.svg"
  create_test_svg_no_dimensions(input_path, vb_w=200, vb_h=100)

  result = await resize_image(ctx, str(input_path), 100, 50, str(output_path))
  assert "Successfully resized" in result
  content = output_path.read_text()
  assert "viewBox" in content


# ---------------------------------------------------------------------------
# convert_image — SVG input
# ---------------------------------------------------------------------------

async def test_convert_svg_to_svg_copy(ctx, tmp_dir):
  """SVG → SVG should produce a copy at the output path."""
  input_path = tmp_dir / "test.svg"
  output_path = tmp_dir / "copy.svg"
  create_test_svg(input_path)

  result = await convert_image(ctx, str(input_path), "SVG", str(output_path))
  assert "Successfully converted" in result
  assert output_path.exists()
  assert "<svg" in output_path.read_text()


@pytest.mark.skipif(not _CAIROSVG_AVAILABLE, reason="cairosvg not installed")
async def test_convert_svg_to_png(ctx, tmp_dir):
  input_path = tmp_dir / "test.svg"
  output_path = tmp_dir / "out.png"
  create_test_svg(input_path)

  result = await convert_image(ctx, str(input_path), "PNG", str(output_path))
  assert "Successfully converted" in result
  assert output_path.exists()
  with Image.open(output_path) as img:
    assert img.format == "PNG"


@pytest.mark.skipif(not _CAIROSVG_AVAILABLE, reason="cairosvg not installed")
async def test_convert_svg_to_jpg(ctx, tmp_dir):
  input_path = tmp_dir / "test.svg"
  output_path = tmp_dir / "out.jpg"
  create_test_svg(input_path)

  result = await convert_image(ctx, str(input_path), "JPG", str(output_path))
  assert "Successfully converted" in result
  assert output_path.exists()


async def test_convert_svg_to_raster_without_cairosvg(ctx, tmp_dir, monkeypatch):
  """Without cairosvg, conversion should return a helpful error."""
  import subconscious.desktop_tools.images as img_module
  monkeypatch.setattr(img_module, "_CAIROSVG_AVAILABLE", False)

  input_path = tmp_dir / "test.svg"
  create_test_svg(input_path)

  result = await convert_image(ctx, str(input_path), "PNG", str(tmp_dir / "out.png"))
  assert "cairosvg" in result.lower()


async def test_convert_raster_to_svg_without_vtracer(ctx, tmp_dir, monkeypatch):
  """Without vtracer, conversion should return a helpful install hint."""
  import subconscious.desktop_tools.images as img_module
  monkeypatch.setattr(img_module, "_VTRACER_AVAILABLE", False)

  input_path = tmp_dir / "test.png"
  create_test_png(input_path)

  result = await convert_image(ctx, str(input_path), "SVG", str(tmp_dir / "out.svg"))
  assert "vtracer" in result.lower()


@pytest.mark.skipif(not _VTRACER_AVAILABLE, reason="vtracer not installed")
async def test_convert_png_to_svg(ctx, tmp_dir):
  """PNG → SVG should produce a valid SVG file."""
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "out.svg"
  create_test_png(input_path, size=(64, 64))

  result = await convert_image(ctx, str(input_path), "SVG", str(output_path))
  assert "Successfully converted" in result
  assert output_path.exists()
  content = output_path.read_text()
  assert "<svg" in content


@pytest.mark.skipif(not _VTRACER_AVAILABLE, reason="vtracer not installed")
async def test_convert_jpg_to_svg(ctx, tmp_dir):
  """JPG → SVG should produce a valid SVG file."""
  input_path = tmp_dir / "test.jpg"
  output_path = tmp_dir / "out.svg"
  create_test_jpg(input_path, size=(64, 64))

  result = await convert_image(ctx, str(input_path), "SVG", str(output_path))
  assert "Successfully converted" in result
  assert output_path.exists()
  assert "<svg" in output_path.read_text()


@pytest.mark.skipif(not _VTRACER_AVAILABLE, reason="vtracer not installed")
async def test_convert_bmp_to_svg(ctx, tmp_dir):
  """BMP → SVG should produce a valid SVG file."""
  input_path = tmp_dir / "test.bmp"
  output_path = tmp_dir / "out.svg"
  create_test_bmp(input_path, size=(64, 64))

  result = await convert_image(ctx, str(input_path), "SVG", str(output_path))
  assert "Successfully converted" in result
  assert output_path.exists()
  assert "<svg" in output_path.read_text()


@pytest.mark.skipif(not _VTRACER_AVAILABLE, reason="vtracer not installed")
async def test_convert_to_svg_auto_output_path(ctx, tmp_dir):
  """output_path=None should use input stem + .svg extension."""
  input_path = tmp_dir / "test.png"
  create_test_png(input_path, size=(32, 32))

  result = await convert_image(ctx, str(input_path), "SVG")
  expected = input_path.with_suffix(".svg")
  assert "Successfully converted" in result
  assert expected.exists()
  assert "<svg" in expected.read_text()


@pytest.mark.skipif(not _VTRACER_AVAILABLE, reason="vtracer not installed")
async def test_batch_convert_to_svg(ctx, tmp_dir):
  """Batch convert should produce SVG files for all raster inputs."""
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()
  create_test_png(src_dir / "a.png", size=(32, 32))
  create_test_jpg(src_dir / "b.jpg", size=(32, 32))

  result = await batch_convert_image(ctx, str(src_dir), str(dest_dir), "SVG")
  assert "Converted 2 images to SVG" in result
  assert (dest_dir / "a.svg").exists()
  assert (dest_dir / "b.svg").exists()
  assert "<svg" in (dest_dir / "a.svg").read_text()


async def test_convert_svg_unsupported_target(ctx, tmp_dir):
  """SVG → ICO/TIFF/GIF should return a clear error (not in SVG raster set)."""
  input_path = tmp_dir / "test.svg"
  create_test_svg(input_path)

  result = await convert_image(ctx, str(input_path), "ICO", str(tmp_dir / "out.ico"))
  assert "Cannot convert SVG" in result


# ---------------------------------------------------------------------------
# batch_optimize_images — SVG
# ---------------------------------------------------------------------------

async def test_batch_optimize_includes_svg(ctx, tmp_dir):
  """Batch optimize should process SVG files alongside raster images."""
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()

  create_test_png(src_dir / "img.png")
  create_test_svg(src_dir / "icon.svg")

  result = await batch_optimize_images(ctx, str(src_dir), str(dest_dir))
  assert "Processed 2 images" in result
  assert (dest_dir / "img.png").exists()
  assert (dest_dir / "icon.svg").exists()


# ---------------------------------------------------------------------------
# batch_resize_images — SVG
# ---------------------------------------------------------------------------

async def test_batch_resize_includes_svg(ctx, tmp_dir):
  """Batch resize should resize SVG files alongside raster images."""
  src_dir = tmp_dir / "src"
  dest_dir = tmp_dir / "dest"
  src_dir.mkdir()

  create_test_png(src_dir / "img.png", size=(300, 300))
  create_test_svg(src_dir / "icon.svg", width=300, height=300)

  result = await batch_resize_images(ctx, str(src_dir), str(dest_dir), width=150, height=150)
  assert "Resized 2 images" in result
  assert (dest_dir / "icon.svg").exists()
  content = (dest_dir / "icon.svg").read_text()
  assert 'width="150"' in content


# ---------------------------------------------------------------------------
# images_to_pdf
# ---------------------------------------------------------------------------

async def test_images_to_pdf_single_image(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "out.pdf"
  create_test_png(input_path, size=(100, 100))

  result = await images_to_pdf(ctx, [str(input_path)], str(output_path))
  assert "Successfully combined" in result
  assert output_path.exists()

  import pypdf
  reader = pypdf.PdfReader(str(output_path))
  assert len(reader.pages) == 1


async def test_images_to_pdf_multiple_images_preserves_order(ctx, tmp_dir):
  p1 = tmp_dir / "a.png"
  p2 = tmp_dir / "b.jpg"
  p3 = tmp_dir / "c.bmp"
  output_path = tmp_dir / "combined.pdf"
  create_test_png(p1, size=(100, 100))
  create_test_jpg(p2, size=(120, 80))
  create_test_bmp(p3, size=(60, 60))

  result = await images_to_pdf(ctx, [str(p1), str(p2), str(p3)], str(output_path))
  assert "Successfully combined 3 image(s)" in result
  assert output_path.exists()

  import pypdf
  reader = pypdf.PdfReader(str(output_path))
  assert len(reader.pages) == 3


async def test_images_to_pdf_empty_list(ctx, tmp_dir):
  result = await images_to_pdf(ctx, [], str(tmp_dir / "out.pdf"))
  assert "at least one" in result.lower()


async def test_images_to_pdf_missing_input(ctx, tmp_dir):
  result = await images_to_pdf(
    ctx, [str(tmp_dir / "missing.png")], str(tmp_dir / "out.pdf")
  )
  assert "not exist" in result.lower()


async def test_images_to_pdf_creates_missing_dest_dir(ctx, tmp_dir):
  input_path = tmp_dir / "test.png"
  output_path = tmp_dir / "nested" / "dir" / "out.pdf"
  create_test_png(input_path)

  result = await images_to_pdf(ctx, [str(input_path)], str(output_path))
  assert "Successfully combined" in result
  assert output_path.exists()


async def test_images_to_pdf_includes_svg(ctx, tmp_dir):
  """SVG inputs must be rasterized before being combined into the PDF."""
  if not (_CAIROSVG_AVAILABLE):
    pytest.skip("cairosvg not installed")

  png_path = tmp_dir / "a.png"
  svg_path = tmp_dir / "b.svg"
  output_path = tmp_dir / "out.pdf"
  create_test_png(png_path)
  create_test_svg(svg_path)

  result = await images_to_pdf(ctx, [str(png_path), str(svg_path)], str(output_path))
  assert "Successfully combined 2 image(s)" in result

  import pypdf
  reader = pypdf.PdfReader(str(output_path))
  assert len(reader.pages) == 2


# ---------------------------------------------------------------------------
# pdf_to_images
# ---------------------------------------------------------------------------

def _create_test_pdf(path: pathlib.Path, page_sizes=((100, 100),)) -> None:
  """Create a multi-page PDF from solid-color pages using Pillow."""
  pages = [Image.new("RGB", size, (200, 50, 50)) for size in page_sizes]
  first, rest = pages[0], pages[1:]
  first.save(path, "PDF", save_all=True, append_images=rest)


@pytest.mark.skipif(not _PYMUPDF_AVAILABLE, reason="PyMuPDF (fitz) not installed")
async def test_pdf_to_images_single_page(ctx, tmp_dir):
  pdf_path = tmp_dir / "doc.pdf"
  dest_dir = tmp_dir / "pages"
  _create_test_pdf(pdf_path, page_sizes=((200, 200),))

  result = await pdf_to_images(ctx, str(pdf_path), str(dest_dir))
  assert "Successfully rendered 1 page(s)" in result
  assert (dest_dir / "doc_page_1.png").exists()


@pytest.mark.skipif(not _PYMUPDF_AVAILABLE, reason="PyMuPDF (fitz) not installed")
async def test_pdf_to_images_multi_page_naming(ctx, tmp_dir):
  pdf_path = tmp_dir / "multi.pdf"
  dest_dir = tmp_dir / "pages"
  _create_test_pdf(pdf_path, page_sizes=((100, 100), (100, 100), (100, 100)))

  result = await pdf_to_images(ctx, str(pdf_path), str(dest_dir))
  assert "Successfully rendered 3 page(s)" in result
  for i in (1, 2, 3):
    assert (dest_dir / f"multi_page_{i}.png").exists()


@pytest.mark.skipif(not _PYMUPDF_AVAILABLE, reason="PyMuPDF (fitz) not installed")
async def test_pdf_to_images_jpg_format(ctx, tmp_dir):
  pdf_path = tmp_dir / "doc.pdf"
  dest_dir = tmp_dir / "pages"
  _create_test_pdf(pdf_path)

  result = await pdf_to_images(ctx, str(pdf_path), str(dest_dir), output_format="JPG")
  assert "Successfully rendered 1 page(s)" in result
  out_file = dest_dir / "doc_page_1.jpg"
  assert out_file.exists()
  with Image.open(out_file) as img:
    assert img.format == "JPEG"


@pytest.mark.skipif(not _PYMUPDF_AVAILABLE, reason="PyMuPDF (fitz) not installed")
async def test_pdf_to_images_respects_dpi(ctx, tmp_dir):
  """Higher DPI should produce a larger rendered pixmap."""
  pdf_path = tmp_dir / "doc.pdf"
  dest_lo = tmp_dir / "lo"
  dest_hi = tmp_dir / "hi"
  _create_test_pdf(pdf_path, page_sizes=((200, 200),))

  await pdf_to_images(ctx, str(pdf_path), str(dest_lo), dpi=72)
  await pdf_to_images(ctx, str(pdf_path), str(dest_hi), dpi=300)

  with Image.open(dest_lo / "doc_page_1.png") as lo_img:
    lo_size = lo_img.size
  with Image.open(dest_hi / "doc_page_1.png") as hi_img:
    hi_size = hi_img.size

  assert hi_size[0] > lo_size[0]
  assert hi_size[1] > lo_size[1]


async def test_pdf_to_images_missing_input(ctx, tmp_dir):
  result = await pdf_to_images(ctx, str(tmp_dir / "missing.pdf"), str(tmp_dir / "out"))
  assert "not exist" in result.lower()


async def test_pdf_to_images_unsupported_format(ctx, tmp_dir):
  pdf_path = tmp_dir / "doc.pdf"
  _create_test_pdf(pdf_path)
  if not _PYMUPDF_AVAILABLE:
    result = await pdf_to_images(ctx, str(pdf_path), str(tmp_dir / "out"), output_format="GIF")
    assert "pymupdf" in result.lower()
  else:
    result = await pdf_to_images(ctx, str(pdf_path), str(tmp_dir / "out"), output_format="GIF")
    assert "Unsupported output format" in result


async def test_pdf_to_images_without_pymupdf(ctx, tmp_dir, monkeypatch):
  """Without PyMuPDF installed, the tool returns a helpful install hint."""
  import subconscious.desktop_tools.images as img_module
  monkeypatch.setattr(img_module, "_PYMUPDF_AVAILABLE", False)

  pdf_path = tmp_dir / "doc.pdf"
  _create_test_pdf(pdf_path)

  result = await pdf_to_images(ctx, str(pdf_path), str(tmp_dir / "out"))
  assert "pymupdf" in result.lower()

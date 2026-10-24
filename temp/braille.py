""" Braille generator """
import sys
import argparse
from PIL import Image
from pathlib import Path


r = "⠀⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿⡀⡁⡂⡃⡄⡅⡆⡇⡈⡉⡊⡋⡌⡍⡎⡏⡐⡑⡒⡓⡔⡕⡖⡗⡘⡙⡚⡛⡜⡝⡞⡟⡠⡡⡢⡣⡤⡥⡦⡧⡨⡩⡪⡫⡬⡭⡮⡯⡰⡱⡲⡳⡴⡵⡶⡷⡸⡹⡺⡻⡼⡽⡾⡿⢀⢁⢂⢃⢄⢅⢆⢇⢈⢉⢊⢋⢌⢍⢎⢏⢐⢑⢒⢓⢔⢕⢖⢗⢘⢙⢚⢛⢜⢝⢞⢟⢠⢡⢢⢣⢤⢥⢦⢧⢨⢩⢪⢫⢬⢭⢮⢯⢰⢱⢲⢳⢴⢵⢶⢷⢸⢹⢺⢻⢼⢽⢾⢿⣀⣁⣂⣃⣄⣅⣆⣇⣈⣉⣊⣋⣌⣍⣎⣏⣐⣑⣒⣓⣔⣕⣖⣗⣘⣙⣚⣛⣜⣝⣞⣟⣠⣡⣢⣣⣤⣥⣦⣧⣨⣩⣪⣫⣬⣭⣮⣯⣰⣱⣲⣳⣴⣵⣶⣷⣸⣹⣺⣻⣼⣽⣾⣿"

def image_to_braille(image_path, width=80, invert=False, threshold=128):
    """Converts a PNG image to Braille art."""
    try:
        img = Image.open(image_path)
        
        # If image has an alpha channel, we use it for the silhouette
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Extract alpha channel to use as the dot source
            alpha = img.convert("RGBA").getchannel('A')
            img = alpha
        else:
            img = img.convert("L")
            
        if invert:
            from PIL import ImageOps
            img = ImageOps.invert(img)
            
    except Exception as e:
        return f"Error opening image: {e}"

    # Calculate target dimensions
    w, h = img.size
    aspect_ratio = h / w
    height = int(width * 2 * aspect_ratio / 4)
    
    img = img.resize((width * 2, height * 4), Image.Resampling.LANCZOS)
    # Convert to monochrome based on threshold
    img = img.point(lambda p: 1 if p > threshold else 0, mode='1')
    pixels = img.load()
    
    output = []
    for y in range(0, height * 4, 4):
        line = ""
        for x in range(0, width * 2, 2):
            idx = 0
            if pixels[x, y]:     idx |= 0x1
            if pixels[x, y+1]:   idx |= 0x2
            if pixels[x, y+2]:   idx |= 0x4
            if pixels[x+1, y]:   idx |= 0x8
            if pixels[x+1, y+1]: idx |= 0x10
            if pixels[x+1, y+2]: idx |= 0x20
            if pixels[x, y+3]:   idx |= 0x40
            if pixels[x+1, y+3]: idx |= 0x80
            line += r[idx]
        output.append(line)
    
    return "\n".join(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to Braille art.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("width", type=int, nargs="?", default=80, help="Width in Braille characters")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write Braille art to this UTF-8 text file instead of the terminal",
    )
    parser.add_argument("--invert", action="store_true", help="Invert the output")
    parser.add_argument("--threshold", type=int, default=128, help="Luminance/Alpha threshold (0-255)")

    args = parser.parse_args()
    braille_art = image_to_braille(args.image, args.width, args.invert, args.threshold)

    if args.output is None:
        print(braille_art)
    else:
        try:
            args.output.write_text(f"{braille_art}\n", encoding="utf-8", newline="\n")
        except OSError as error:
            parser.error(f"Could not write output file '{args.output}': {error}")
        print(f"Wrote Braille art to {args.output}", file=sys.stderr)

from PIL import Image, ImageDraw, ImageFont


WIDTH = 700
HEIGHT = 250

image = Image.new(
    "RGBA",
    (WIDTH, HEIGHT),
    (0, 0, 0, 0)
)

draw = ImageDraw.Draw(image)


# Try to use a Windows font.
font_path = r"C:\Windows\Fonts\arialbd.ttf"

try:
    font = ImageFont.truetype(font_path, 110)
except:
    font = ImageFont.load_default()


text = "FAHH!"


# Get text dimensions.
bbox = draw.textbbox((0, 0), text, font=font)

text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]


x = (WIDTH - text_width) // 2
y = (HEIGHT - text_height) // 2


# Black outline.
for dx in range(-5, 6):
    for dy in range(-5, 6):

        draw.text(
            (x + dx, y + dy),
            text,
            font=font,
            fill=(0, 0, 0, 255)
        )


# Main text.
draw.text(
    (x, y),
    text,
    font=font,
    fill=(255, 255, 255, 255)
)


image.save(
    "assets/fahh.png"
)

print("Created assets/fahh.png")
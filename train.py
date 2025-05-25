from PIL import Image
img = Image.open("archive/stockimg.jpg")
img.save("bg.png", "PNG")
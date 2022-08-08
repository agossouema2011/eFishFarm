import glob
import os
from PIL import Image

dst_dir = 'Ich_Diseased_Fish32x32'
os.makedirs(dst_dir, exist_ok=True)
files = glob.glob('Ich_Diseased_Fish/*')

for f in files:
    root, ext = os.path.splitext(f)
    if ext in ['.jpg', '.jpeg','.png']:
        img = Image.open(f)
        img_resize = img.resize((32, 32))
        #print(img_resize)
        basename = os.path.basename(root)
        img_resize.save(os.path.join(dst_dir,basename + ext))

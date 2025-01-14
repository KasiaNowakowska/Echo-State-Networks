"""
python script to generate gif

Usage: makegif.py <input_path> <output_gif> [--duration=<duration]

Options:
    --duration=<duration>    duration of each fram in gif in seconds [default: 5]

"""

from PIL import Image
import os
from docopt import docopt
args = docopt(__doc__)


def create_gif(input_path, output_gif, duration):
    images = []
    
    # List all files in the image folder
    files = sorted(os.listdir(input_path))
    print(files)
    
    for file_name in files:
        # Load each image
        if file_name.endswith('.png'):
            file_path = os.path.join(input_path, file_name)
            try:
                # Open the image file
                img = Image.open(file_path)
                
                # Resize the image (optional)
                # img = img.resize((width, height))
                
                # Append image to the list
                images.append(img)
            except IOError:
                print(f"Unable to load image: {file_path}")
    
    # Save images as a GIF
    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF created: {output_gif}")
    else:
        print("No images found in the specified directory.")


# Extract arguments
input_dir = args['<input_path>']
output_gif = args['<output_gif>']
duration = float(args['--duration'])

#if not os.path.exists(output_path):
 #   os.makedirs(output_path)
  #  print('made directory')
    
# Create GIF from images in the specified directory
create_gif(input_dir, output_gif, duration)

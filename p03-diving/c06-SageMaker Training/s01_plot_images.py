import glob

# +
import ipyplot

training_samples = glob.glob(f"tmp/train/*/*.png")
len(training_samples)

for i in range(0, 10):
    image_files = glob.glob(f"tmp/train/{i}/*.png")
    print(f'---{i}---')
    ipyplot.plot_images(image_files, max_images=5, img_width=128)

from PIL import Image
from got10k.datasets import GOT10k
from got10k.utils.viz import show_frame

dataset = GOT10k(root_dir='data/GOT-10k', subset='train')


# indexing
img_file, anno = dataset[10]

# for-loop
for s, (img_files, anno) in enumerate(dataset):
    seq_name = dataset.seq_names[s]
    print('Sequence:', seq_name)

    # show all frames
    for f, img_file in enumerate(img_files):
        image = Image.open(img_file)
        show_frame(image, anno[f, :])
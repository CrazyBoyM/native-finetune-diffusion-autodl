# from AUTOMATC1111
# maybe modified by Nyanko Lepsoni
# modified by crosstyan
import os.path
import re
import tempfile
import argparse
import glob
import zipfile
import numpy as np

from basicsr.utils.download_util import load_file_from_url
from PIL import Image
import tqdm

import torch
# import gradio as gr

import deep_danbooru_model

print("Loading model...")
script_path = os.path.realpath(__file__)
model_path = os.path.join(os.path.dirname(script_path), "deepdanbooru-model/model-resnet_custom_v3.pt")
model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load(model_path))

model.eval()
model.half()
model.cuda()
print("Model loaded.")

re_special = re.compile(r"([\\()])")

def get_deepbooru_tags_from_model(
    image,
    threshold,
    alpha_sort=False,
    use_spaces=True,
    use_escape=True,
    include_ranks=False,
):
    image = image.resize((512, 512))
    a = np.expand_dims(np.array(image, dtype=np.float32), 0) / 255

    with torch.no_grad(), torch.autocast("cuda"):
        x = torch.from_numpy(a).cuda()

        # first run
        y = model(x)[0].detach().cpu().numpy()

        # measure performance
        for n in range(10):
            model(x)

    unsorted_tags_in_theshold = []
    for i, p in enumerate(y):
        if p >= threshold:
            tag = model.tags[i]
            if tag.startswith("rating:"):
                continue
            
            unsorted_tags_in_theshold.append((p, tag))

    # sort tags
    result_tags_out = []
    sort_ndx = 0
    if alpha_sort:
        sort_ndx = 1

    # sort by reverse by likelihood and normal for alpha, and format tag text as requested
    unsorted_tags_in_theshold.sort(key=lambda y: y[sort_ndx], reverse=(not alpha_sort))
    for weight, tag in unsorted_tags_in_theshold:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace("_", " ")
        if use_escape:
            tag_outformat = re.sub(re_special, r"\\\1", tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{weight:.3f})"

        result_tags_out.append(tag_outformat)

    # print("\n".join(sorted(result_tags_print, reverse=True)))

    return ", ".join(result_tags_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--alpha_sort", type=bool, default=False)
    parser.add_argument("--use_spaces", type=bool, default=True)
    parser.add_argument("--use_escape", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--include_ranks", type=bool, default=False)

    args = parser.parse_args()

    types = ('*.jpg', '*.png', '*.jpeg', '*.gif', '*.webp', '*.bmp') 
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(os.path.join(args.path, files)))
        # print(glob.glob(args.path + files))
        
    for image_path in tqdm.tqdm(files_grabbed, desc="Processing"):
        image = Image.open(image_path).convert("RGB")
        prompt = get_deepbooru_tags_from_model(
            image,
            args.threshold,
            alpha_sort=args.alpha_sort,
            use_spaces=args.use_spaces,
            use_escape=args.use_escape,
            include_ranks=args.include_ranks,
        )
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = os.path.join(args.path, f"{image_name}.txt")
        # print(f"writing {txt_filename}: {prompt}")
        with open(txt_filename, 'w') as f:
            f.write(prompt)
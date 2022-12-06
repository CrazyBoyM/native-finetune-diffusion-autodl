# このスクリプトのライセンスは、Apache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

import argparse
import glob
import os
import json

from PIL import Image
from tqdm import tqdm

from clip_interrogator import Interrogator, Config


print("Loading model...")
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
print("Model loaded.")

def main(args):
    image_paths = glob.glob(os.path.join(args.train_data_dir, "*.jpg")) + glob.glob(os.path.join(args.train_data_dir, "*.png"))
    print(f"found {len(image_paths)} images.")

    for image_path in tqdm(image_paths):
        raw_image = Image.open(image_path)
        if raw_image.mode != "RGB":
            print(f"convert image mode {raw_image.mode} to RGB: {image_path}")
            raw_image = raw_image.convert("RGB")

        caption = ci.interrogate(raw_image)
      
        with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding='utf-8') as f:
            f.write(caption + "\n")
            if args.debug:
                print(image_path, caption)    

    print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  # parser.add_argument("caption_weights", type=str,
  #                     help="BLIP caption weights (model_large_caption.pth) / BLIP captionの重みファイル(model_large_caption.pth)")
  parser.add_argument("--caption_extention", type=str, default=None,
                      help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
  parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / 出力されるキャプションファイルの拡張子")
  parser.add_argument("--beam_search", action="store_true",
                      help="use beam search (default Nucleus sampling) / beam searchを使う（このオプション未指定時はNucleus sampling）")
  parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
  parser.add_argument("--num_beams", type=int, default=1, help="num of beams in beam search /beam search時のビーム数（多いと精度が上がるが時間がかかる）")
  parser.add_argument("--top_p", type=float, default=0.9, help="top_p in Nucleus sampling / Nucleus sampling時のtop_p")
  parser.add_argument("--max_length", type=int, default=75, help="max length of caption / captionの最大長")
  parser.add_argument("--min_length", type=int, default=5, help="min length of caption / captionの最小長")
  parser.add_argument("--debug", action="store_true", help="debug mode")

  args = parser.parse_args()

  # スペルミスしていたオプションを復元する
  if args.caption_extention is not None:
    args.caption_extension = args.caption_extention

  main(args)

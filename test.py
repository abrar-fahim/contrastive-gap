from PIL import Image
import requests
import clip

from hf_clip import HFClip

image2 = Image.open(requests.get('https://m.media-amazon.com/images/M/MV5BMTM3OTUwMDYwNl5BMl5BanBnXkFtZTcwNTUyNzc3Nw@@._V1_FMjpg_UX1000_.jpg', stream=True).raw)



captions = ['face of scarlett johansson', 'face of Sophie Turner']


device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


img2dataset --url_list laion400m_0-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion400m-data --processes_count 16 --thread_count 128 --image_size 256\
             --save_additional_columns '["NSFW","similarity","LICENSE"]'
import gdown, sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("link", type=str, help="Publicly accessible link to file in drive")
parser.add_argument("--dest", type=str, default="./", help="Destination directory")
args = parser.parse_args()

url = args.link
dest = args.dest
file_id = url.split("/")[-2]
prefix = "https://drive.google.com/uc?/export=download&id="
os.chdir(dest)
gdown.download(prefix+file_id)
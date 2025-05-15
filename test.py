import requests
from io import BytesIO
from pathlib import Path
import os
import time
import pprint

url = "http://127.0.0.1:6759/gui/agent"

def convert_file(file, filename=None):
    if isinstance(file, bytes):  # raw bytes
        file = BytesIO(file)
    elif hasattr(file, "read"):  # a file io like object
        filename = filename or file.name
    else:  # a local path
        file = Path(file).absolute().open("rb")
        filename = filename or os.path.split(file.name)[-1]
    return filename, file

def test_post(image_path, instruction):
    _, file = convert_file(image_path)
    start = time.time()
    data = {"instruction": instruction}
    response = requests.post(url=url,
                            files={"image_data":file},
                            data=data)
    print(response.json())
    print("use time:", time.time() - start)

if __name__ == "__main__":
    image_path = "./assets/Snipaste1.png"
    instruction = "去哔哩哔哩看李子柒的最新视频，并且点赞"
    test_post(image_path, instruction)
from tqdm import tqdm
import requests
import os
import tempfile


def download(ckpt_dir: str, url: str):
    name = url[url.rfind("/") + 1 : url.rfind("?")]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, 'flaxmodels')
    ckpt_file = os.path.join(ckpt_dir, name)
    
    if not os.path.exists(ckpt_file):
        print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    prog_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    ckpt_file_temp = os.pardir.join(ckpt_dir, name + ".temp")
    with open(ckpt_file_temp, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            prog_bar.update(len(chunk))
            f.write(chunk)
    prog_bar.close()

    if total_size_in_bytes != 0 and prog_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
        if os.path.exists(ckpt_file_temp):
            os.remove(ckpt_file_temp)
    else:
        os.rename(ckpt_file_temp, ckpt_file)

    return ckpt_file
import os
import shutil
import gdown


def main():
    os.makedirs('weights', exist_ok=True)
    files = {
        'weights/craft_mlt_25k.pth':
            'https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view',
        'weights/craft_refiner_CTW1500.pth':
            'https://drive.google.com/file/d/1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO/view',
    }
    for output, url in files.items():
        if os.path.exists(output):
            print('😪  Found %s downloaded already' % (output))
        else:
            print('🏃  Downloading %s from %s' % (output, url))
            download(url, output)
    compressed_folders = {
        'weights/best_layoutlmv3_20241221.zip':
            'https://drive.google.com/file/d/1WyGOg5PAwn_N4ppVsiPAgIr45pJtK0cA/view',
        'weights/best_phobert_20241229.zip':
            'https://drive.google.com/file/d/17MyYOkyoU6xmUDT8h3MEUuyAYRV4wfYb/view',   
    }
    for name, url in compressed_folders.items():
        output, ext = os.path.splitext(name)
        if os.path.exists(output):
            print('😪  Found %s downloaded already' % (output))
        else:
            print('🏃  Downloading %s from %s' % (name, url))
            download(url, name)
            print('📦  Uncompressing %s' % name)
            shutil.unpack_archive(name, output)
            print('🧹  Deleting %s' % name)
            try:
                os.remove(name)
            except:
                pass
    print('🫡  ALL DONE')


def download(url, output):
    parent = os.path.split(output)[0]
    os.makedirs(parent, exist_ok=True)
    gdown.download(url=url, output=output, fuzzy=True)


if __name__ == '__main__':
    main()

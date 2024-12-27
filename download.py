import os
import gdown


def main():
    files = {
        'weights/craft_mlt_25k.pth':
            'https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view',
        'weights/craft_refiner_CTW1500.pth':
            'https://drive.google.com/file/d/1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO/view',
    }
    for output, url in files.items():
        if os.path.exists(output):
            print('😪 Found %s downloaded already' % (output))
        else:
            print('🏃‍♂️‍➡️ Downloading %s from %s' % (output, url))
            download(url, output)
    print('🫡 ALL DONE')


def download(url, output):
    parent = os.path.split(output)[0]
    os.makedirs(parent, exist_ok=True)
    gdown.download(url=url, output=output, fuzzy=True)


if __name__ == '__main__':
    main()

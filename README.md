# Vietnamese Text Recognition Application

**This project is under construction.**

I used Python 3.10.11 and installed the packages as described in the
`requirements.txt` file. Any other version of Python or the packages
are untested.

```mermaid
graph TD;

image[Input image] --> detection{Text
    Detection}

detection --> bboxes[Bounding boxes
    of texts]

image --> cropper{Image
    Cropper}

bboxes --> cropper

cropper --> pieces[Pieces of image]

pieces --> ocr{OCR}

ocr --> result[Texts]
```

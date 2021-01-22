from io import BytesIO
from pandas.core import base
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile, Form
from starlette.responses import Response
from PIL import Image
import base64
import json
from inference import tryon
import matplotlib.pyplot as plt

app = FastAPI()

@app.post('/tryon')
def get_tryon_image(image: UploadFile = File(...),
        image_parse: UploadFile = File(...),
        cloth: UploadFile = File(...),
        cloth_mask: UploadFile = File(...),
        pose: UploadFile = File(...)):
    img = Image.open(image.file)
    img_parse = Image.open(image_parse.file)
    clo = Image.open(cloth.file)
    clo_mask = Image.open(cloth_mask.file)
    po = json.load(pose.file)
    output = tryon(img, img_parse, clo, clo_mask, po)
    output_image = Image.fromarray(output)
    buffered = BytesIO()
    output_image.save(buffered, format='JPEG')
    return Response(buffered.getvalue(), media_type='image/jpeg')

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8081)
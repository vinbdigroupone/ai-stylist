from io import BytesIO
import json
from retrieval_module.retriever import retrieve
from tryon_module.inference import tryon
from recommendation_module.Recommender.engine import generate_outfit
from recommendation_module.Recommender.build import build_system
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile, Form
from starlette.responses import JSONResponse, Response
from PIL import Image
import base64

app = FastAPI()

@app.post('/imagesearch')
def retrieve_image(file: UploadFile = File(...), k: int = Form(...)):
    image = Image.open(file.file)
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    image_str = base64.b64encode(buffered.getvalue())
    results = retrieve(image_str, True, k)
    return JSONResponse(status_code=200, content=results)

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

@app.post('/outfitrecommender')
def get_recommendation(image: UploadFile = File(...)):
    engine, model, new_type_spaces, gpu = build_system()
    img = Image.open(image.file)
    results = generate_outfit(img, 'tops', engine, model, new_type_spaces, gpu)
    print(results)
    return Response()

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8080)
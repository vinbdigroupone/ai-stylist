from fastapi.testclient import TestClient
import base64
from io import BytesIO
from PIL import Image
import uvicorn

from fastapi import FastAPI
# from api import app

from Recommender.engine import generate_outfit
from Recommender.build import build_system

app = FastAPI()

@app.post('/outfitrecommender')
def test_post_api():
    # engine, model, new_type_spaces, gpu = build_system()

    # img_path = './test_images/2.jpg'
    img_path = './test_images/2'
    results = generate_outfit(img_path, item_type, \
                                engine, model, new_type_spaces, gpu=gpu)
    
    res = client.get('/image_search',
                       json={'image_str': image_str, 'num_results': k})

    return JSONResponse(status_code=200, content=results)

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8884)
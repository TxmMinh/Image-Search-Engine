import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
from io import BytesIO
from FashionImgSearch_Predictions import GenerateSimilarImages
import logging
import streamlit as st
import requests

app = FastAPI()

# Mount the 'static' directory to serve CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates configuration
templates = Jinja2Templates(directory="templates")

# Function to load the query image
def load_input_url(input_url):
    response = requests.get(input_url)
    queryImage = Image.open(BytesIO(response.content))
    st.image(queryImage)

    return queryImage

# Generate top 8 similar images
def return_similar_images(query_image):
    sim_img_gen = GenerateSimilarImages(query_image)
    _, top_8_images_path, top_8_images_desc = sim_img_gen.generate_similar_images()

    return [
        top_8_images_path[i] for i in range(8)
    ], [
        top_8_images_desc[i] for i in range(8)
    ]

# Main endpoint
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Search endpoint
@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query_url: str = Form(...)):
    try:
        query_image = load_input_url(query_url)
    except:
        return templates.TemplateResponse("index.html", {"request": request, "result": "Invalid URL"})

    # Get the FastAPI logger
    logger = logging.getLogger(__name__)

    # Log structured information
    logger.info("Received search request", extra={"query_url": query_url, "query_image": query_image})

    # Display the spinner
    # with st.spinner('Wait for it...'):
    #     time.sleep(40)

    top_8_images, top_8_descriptions = return_similar_images(query_image)

    result_data = {
        "query_image": query_url,
        "top_8_images": top_8_images,
        "top_8_descriptions": top_8_descriptions
    }

    return templates.TemplateResponse("index.html", {"request": request, "result": result_data})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.2", port=8000)

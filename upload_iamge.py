import io
from PIL import Image
def upload_image():
    from google.colab import files
    # Upload an image file
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    image = Image.open(io.BytesIO(uploaded[file_name]))
    return image

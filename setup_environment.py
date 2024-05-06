def setup_environment() :
    !pip install git+https://github.com/huggingface/transformers.git@main
    !pip install -q datasets

    from google.colab import drive
    drive.mount('/content/drive') 
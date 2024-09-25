# How to Run the Project
1. Open the mias.ipnyb in Google Colab.
2. Download or pull the required images from Docker: docker pull miep-it66-2019
2. Upload the images from the data/images folder to the session storage in Google Colab, inside the `content/data/` directory.
3. Build the Docker image(if not already built):
docker build -t miep-it66-2019
4. Run the Docker container:
docker run -v /content/data:/data miep-it66-2019

Example:

1. Pull the Docker image from Docker Hub:
	docker pull your-docker-image
2. Mount the model directory and run the container:
	docker run -v /content/models:/app/models miep-it66-2019

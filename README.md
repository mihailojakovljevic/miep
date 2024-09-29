# How to Run the Project
1. Pull the Docker image from Docker Hub:
	docker pull mihailojakovljevic/miep-it66-2019:latest
2. Mount the model directory and run the container:
	docker run -v /content/models:/app/models miep-it66-2019

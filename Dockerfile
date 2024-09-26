FROM python:3.10

# Set working directory in container
WORKDIR /app

RUN mkdir -p /model

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code to the container
COPY . .

# Optional: Set default command to open the script
CMD ["python", "inference.py"]

FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy your code to the container
COPY . .

# Optional: Set default command to open the script (but in Colab you run it manually)
CMD ["python", "mias.ipynb"]

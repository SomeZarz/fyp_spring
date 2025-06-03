@echo off
REM Build the Docker image
docker build -t fyp-image .

REM Run the Docker container
docker run --name fyp-container -d -v "%cd%\src:/code/src" -v "%cd%\data:/code/data" fyp-image

echo Script finished.

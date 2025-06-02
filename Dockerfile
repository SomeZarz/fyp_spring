# Base image, Python 3.13
FROM python:3.13-slim-bookworm

# Set environment variables for better Python development
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

# Installing Dependencies
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

CMD [ "sleep", "infinity"]
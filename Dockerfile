FROM python:3.12-alpine
RUN apk add --no-cache rust cargo

WORKDIR /app
ENV TF_USE_LEGACY_KERAS="1"
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
COPY . .
CMD ["fastapi", "dev", "newsfeed.py"]
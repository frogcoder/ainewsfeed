FROM python:3.12

WORKDIR /app
ENV TF_USE_LEGACY_KERAS="1"
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
COPY . .
ENV NLTK_DATA="./nltk_data"
CMD ["fastapi", "run", "serve.py"]
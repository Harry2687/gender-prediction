FROM python:3.12.8-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /gender-prediction

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

RUN python download_model.py

EXPOSE 3000

CMD ["shiny", "run", "app.py", "-h", "0.0.0.0", "-p", "3000"]

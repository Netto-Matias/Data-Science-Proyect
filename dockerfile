FROM python:3.9

WORKDIR /app

COPY ./data /app/data
COPY requirements.txt /app

RUN pip3 install -r requirements.txt

CMD ["uvicorn", "modulos.main:app", "--host= 127.0.0.1", "--port=8000"]
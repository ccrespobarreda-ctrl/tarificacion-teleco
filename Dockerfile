FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir numpy==2.4.3
RUN pip install --no-cache-dir -r requirements.txt

COPY modelo_tarificacion_v1.pkl .
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

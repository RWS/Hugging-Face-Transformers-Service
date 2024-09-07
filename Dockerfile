FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

ENV PORT=8001  
# set a default port to 8001 it not set from the .env file

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/huggingface_ts.py ./src/huggingface_ts.py

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "src.huggingface_ts:app", "--host", "0.0.0.0", "--port", "${PORT}"]
FROM python:3.12 AS builder 

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt 
RUN git clone https://github.com/openai/CLIP.git
WORKDIR /app/CLIP
RUN pip install .
RUN find /usr/local/lib/python3.12/site-packages/ -type d \( -name "tests" -o -name "examples" \) -exec rm -rf {} +
FROM python:3.12-slim AS runtime

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY app.py /app/app.py

EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
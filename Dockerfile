FROM python:3.9-slim

WORKDIR /app

# Copy the Python script into the container
COPY main.py .

# Install dependencies (if any)
RUN pip install -r requirements.txt

CMD ["python3", "main.py"]

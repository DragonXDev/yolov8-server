FROM python:3.10-slim

# Avoid prompts and update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

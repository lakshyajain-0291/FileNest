FROM python:3.12.3-slim
WORKDIR /app
COPY . .
# COPY "/home/xyphoes/.venv" .
RUN "uv venv "
# RUN pip install -r requirements.txt
CMD ["python3", "main.py"]
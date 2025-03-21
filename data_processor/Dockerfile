FROM python:3.9-slim
WORKDIR /app
COPY processor_requirements.txt /app/
RUN pip install --no-cache-dir -r processor_requirements.txt
COPY . /app/
CMD ["python", "data_processor.py"]



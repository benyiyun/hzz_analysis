FROM python:3.9-slim
WORKDIR /app
COPY plotter_requirements.txt /app/
RUN pip install --no-cache-dir -r plotter_requirements.txt
COPY . /app/
CMD ["python", "plotter.py"]

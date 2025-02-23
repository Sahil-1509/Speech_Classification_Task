# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /app/

# Expose any ports if needed (for example, if your Temporal client listens on a port)
EXPOSE 8080

# Optionally, specify the entrypoint (here we assume the Temporal worker should start)
# Adjust this as needed (e.g., you might want to start a worker or a client, or even run a command that starts both in separate containers)
CMD ["python", "temporal_workflow/worker.py"]
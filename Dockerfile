# Use Python 3.10.10 as base image
FROM python:3.10.10-slim

# Add libgomp1 package
RUN apt-get update && apt-get install -y libgomp1

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports for Flask
EXPOSE 5000

# Create a script to run Flask
RUN echo '#!/bin/bash\n\
uvicorn app:app --host 0.0.0.0 --port 8000 & \
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0' > start.sh && \
chmod +x start.sh

# Run the start script
CMD ["./start.sh"]

# Create necessary directories
RUN mkdir -p models data

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501 
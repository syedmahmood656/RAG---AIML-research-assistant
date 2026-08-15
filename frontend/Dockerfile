FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Streamlit default port. Inside docker-compose we map 8501 -> 8501.
EXPOSE 8501

# --server.address=0.0.0.0 is required for the container to be reachable
# outside the loopback interface.
# --browser.gatherUsageStats=false keeps the analytics prompt out of the UI.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false", \
     "--server.headless=true"]

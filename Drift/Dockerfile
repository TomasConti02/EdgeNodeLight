# light python container images
FROM python:3.10-slim


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn numpy scikit-learn requests redis python-multipart httpx

COPY driftv6.py .
#toghter the container image upload also the ood informations
COPY centroids.npy inv_cov_matrix.npy .

CMD ["uvicorn", "driftv6:app", "--host", "0.0.0.0", "--port", "8080"]

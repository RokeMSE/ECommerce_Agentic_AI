FROM python:3.10.7

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Copy the agents directory from the project root into the container
COPY ../../agents /app/agents
COPY . .

# This command keeps the container running for testing purposes
CMD ["/bin/sh", "-c", "python -c \"print('Training service ready...')\" && tail -f /dev/null"]

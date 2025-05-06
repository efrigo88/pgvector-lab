FROM python:3.10-slim

# Set environment variables
ENV PYSPARK_PYTHON=python3

# Install Java 17 and minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    curl \
    procps \
    libpq-dev \
    gcc \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME based on architecture
RUN if [ "$(uname -m)" = "aarch64" ]; then \
    export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64; \
    else \
    export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64; \
    fi && \
    echo "export JAVA_HOME=$JAVA_HOME" >> /etc/profile.d/java.sh

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy only dependency definition first for better layer caching
COPY pyproject.toml .

# Install dependencies using uv without virtual environment
RUN uv pip install --system -e . && rm -rf ~/.cache

# Copy the rest of the application
COPY . .

# The command to run your script
CMD ["python", "-m", "src.main"]

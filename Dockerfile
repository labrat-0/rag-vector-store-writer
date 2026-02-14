# 1. Use a stable, lightweight Python base
FROM python:3.11-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/usr/src/app"

# 3. Set the working directory
WORKDIR /usr/src/app

# 4. Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy source code
COPY . .

# 6. Start the actor
CMD ["python3", "-m", "src.agent.main"]

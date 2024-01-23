# Use a specific Python version
FROM python:3.8

# Create working directory and copy files
WORKDIR /app
COPY . /app
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8080

# Run app
CMD ["streamlit", "run", "--server.port", "8080", "main.py"]

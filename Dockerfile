# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents (where your Python file is) into the container at /app
COPY . /app

# Install any needed dependencies (if you have a requirements.txt)
# RUN pip install -r requirements.txt

# Expose the port Streamlit will run on
# EXPOSE 8501

# Set the default command to run Streamlit``
CMD ["python", "test.py"]
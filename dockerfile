# Use an official Python runtime as a parent image
FROM python:3.8

# Set environment variables
ENV APP_HOME /app
ENV PYTHONUNBUFFERED 1

ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="IsHanGarg/neurips-model"

# Create and set the working directory
WORKDIR $APP_HOME

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Define the command to run your application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

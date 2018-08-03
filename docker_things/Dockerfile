# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install jupyter -U && pip install jupyterlab

# Make port 80 available to the world outside this container
EXPOSE 8888

# Define environment variable
ENV NAME World

# Run app.py when the container launches
ENTRYPOINT ["jupyter", "lab","--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

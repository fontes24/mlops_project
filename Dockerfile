FROM python:3.12-slim 

# Copy the entire project
COPY . mlops_project/

# Install the package with dependencies
RUN pip install ./mlops_project

# Set the working directory
WORKDIR /mlops_project

# Expose the port gunicorn will listen on
EXPOSE 5001

# Run gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:5001", "app.main:app"]
# Use an official Jupyter base image
FROM jupyter/base-notebook

# Copy the Jupyter Notebook file into the container
COPY ["Food Forecasting.ipynb", "./"]

# Install additional dependencies
RUN pip install matplotlib scikit-learn numpy pandas seaborn datetime tensorflow scipy

# Expose the default Jupyter port
EXPOSE 8888

# Start Jupyter Notebook when the container launches
CMD ["start-notebook.sh", "--NotebookApp.token=''"]


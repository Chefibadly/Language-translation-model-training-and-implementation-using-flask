# init a base image (Alpine is small Linux distro)
FROM tensorflow/tensorflow
# define the present working directory
WORKDIR /app
# copy the contents into the working dir
ADD . /app
# run pip to install the dependencies of the flask app
RUN pip install -r requirements.txt
# define the command to start the container
CMD ["python","app.py"]
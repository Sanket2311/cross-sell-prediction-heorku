# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY ./requirements.txt /code

# install dependencies
RUN pip install -r /code/requirements.txt

# copy the content of the local src directory to the working directory
COPY ./ /code

EXPOSE 80/udp

CMD [ "streamlit", "run","app.py","--server.port","80"] 
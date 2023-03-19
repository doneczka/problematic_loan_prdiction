#Base Image to use
FROM python:3.10-slim-bullseye

RUN apt update && apt upgrade -y

#Change Working Directory to app directory
WORKDIR /app

#Expose port 80
EXPOSE 80

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt
RUN pip install --upgrade pip

#Copy all files in current directory into app directory
COPY . /app

#Run the application on port 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
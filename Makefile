DOCKER_IMAGE_NAME=pytorch-transformer-tutorial
DOCKER_CONTAINER_NAME=pytorch-transformer-tutorial-container
PORT=8888
GPU_DEVICE_ID=0
LOCAL_HOST=0.0.0.0
LOCAL_PORT=8888
LOCAL_USER=$(shell whoami)
LOCAL_UID=$(shell id -u $(LOCAL_USER))
LOCAL_GID=$(shell id -g $(LOCAL_USER))


.PHONY: build run run-experiment clean


build: Dockerfile requirements.txt Makefile 
	docker build -t $(DOCKER_IMAGE_NAME) .

	
run: build
	docker run --name $(DOCKER_CONTAINER_NAME) \
               -p $(LOCAL_HOST):$(LOCAL_PORT):$(PORT) \
               --gpus '"device=$(GPU_DEVICE_ID)"' \
               -v /home/$(LOCAL_USER):/home/$(LOCAL_USER) \
               -v /etc/passwd:/etc/passwd:ro \
               -v /etc/group:/etc/group:ro \
               -u $(LOCAL_UID):$(LOCAL_GID) \
               -it \
               --rm \
               $(DOCKER_IMAGE_NAME) bash

               
run-experiment: build
	docker run --name $(DOCKER_CONTAINER_NAME) \
               -p $(LOCAL_HOST):$(LOCAL_PORT):$(PORT) \
               --gpus '"device=$(GPU_DEVICE_ID)"' \
               -v /home/$(LOCAL_USER):/home/$(LOCAL_USER) \
               -v /etc/passwd:/etc/passwd:ro \
               -v /etc/group:/etc/group:ro \
               -u $(LOCAL_UID):$(LOCAL_GID) \
               -it \
               --rm \
               $(DOCKER_IMAGE_NAME) bash -c "python main.py"

               
clean:
	-docker rm -f $(DOCKER_CONTAINER_NAME)
	-docker rmi -f $(DOCKER_IMAGE_NAME)

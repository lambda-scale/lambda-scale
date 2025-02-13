## Multi-thread front-end

### 1. Docker images

#### pull server and client images

```shell
docker login --username=fc_cn@test.aliyunid.com registry.cn-shanghai.aliyuncs.com
password: Serverless123@aliyun

docker pull registry.cn-shanghai.aliyuncs.com/fc-demo2/gpu-swap-standalone-base:client-multi-thread
docker pull registry.cn-shanghai.aliyuncs.com/fc-demo2/gpu-swap-standalone-base:server-multi-thread

```

### 2. Reproduce the GPU memory warnings

Step 1: run the server docker

#### run server

```shell
docker run --gpus all --rm --network=host --ipc=host -v /dev/shm/ipc:/cuda -e MEM_LIMIT_IN_GB=25 -e IO_THREAD_NUM=4 -it  standalone-server  bash start.sh
```

Step 2: run the client docker

`internal.py` accepts one parameter for specifying the desired thread number, for reproducing the warnings, use thread number greater than 2.

In this example, we use 2 threads to reproduce the warnings.

#### run client

```shell
docker run --rm -it --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE --network=host --ipc=host -v /dev/shm/ipc:/cuda --name client-0 client-multi-thread python3 internal.py 2
```
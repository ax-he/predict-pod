# predict-pod

A predictor demo for forecasting resource overhead of prediction tasks

## 特性

### V0.x

可以识别用户上传的视频转码任务所需资源开销

### V1.x

可以识别用户上传的GEMM计算任务所需资源开销

## 部署

**部署k3d集群**

k3d cluster create myk3s

**构建本地镜像**

docker build -t pred-svc:0.8 .

docker build -t blas-probe:0.3 ./ldpreload

**加载到k3d中**
k3d image import pred-svc:0.8 -c myk3s
k3d image import blas-probe:0.3 -c myk3s


**创建RBAC（第一次必须做，后续RBAC文件有改动时也要）**
kubectl apply -f rbac-pred-svc-reader.yaml -n default


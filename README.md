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

`docker build -t pred-svc:2.1 .`

`docker build -f Dockerfile.probe -t blas-probe:0.3 ./ldpreload/gemm`

`docker build -f Dockerfile.probe -t fft-probe:0.1 ./ldpreload/fft`

**检查docker镜像中是否有指定内容**

`docker run --rm -it pred-svc:2.1 bash -lc "cat /app/listen.py | grep '# 9.15 v2.1 test'"  # 例如查找listen.py中是否含有“# 9.15 v2.1 test”内容`

**检查docker镜像中是否含有探针.so**

`docker run --rm -it fft-probe:0.1 bash -lc 'ls -l /opt/probe/lib && ldd /opt/probe/lib/libfftprobe.so || true'`

**在本地测试镜像中的探针是否能使用**

`docker run --rm -it -v $(pwd):/work fft-probe:0.1 bash -lc "`
  `cd /work && \`
  `gcc -O2 fft_test.c -o fft_test -lfftw3 -lfftw3f -lm && \`
  `TIME_BUDGETS='0.02,0.05' IO_FACTOR=2.0 \`
  `LD_PRELOAD=/opt/probe/lib/libfftprobe.so ./fft_test`
`"`

**加载到k3d中**

`k3d image import pred-svc:2.1 -c myk3s`

`k3d image import blas-probe:0.3 -c myk3s`

`k3d image import fft-probe:0.1 -c myk3s`


**创建RBAC（第一次必须做，后续RBAC文件有改动时也要）**

`kubectl apply -f rbac-pred-svc-reader.yaml -n default`

**部署应用配置**

A) apply更新 (推荐)

`kubectl apply -f pred-svc.yaml -n default`

B) 保持YAML不变，切换运行的Deployment

`kubectl -n default set image deploy/pred-svc pred-svc=pred-svc:2.1`

**查看滚动发布状态**

`kubectl -n default rollout status deploy/pred-svc`

## 验证与排错

**看 Pod 与容器是否 Running/Ready**

`kubectl -n default get pods -l app=pred-svc -o wide`

**看启动日志（只有一个容器时不必 -c）**

`kubectl -n default logs -l app=pred-svc --tail=200`

**查看 Service 与 Endpoints（确认 8000 端口被正确指向）**

`kubectl -n default get svc pred-svc -o yaml`

`kubectl -n default get endpoints pred-svc -o wide`

## 本地连通性测试

**把本机 8080 转发到集群内 pred-svc 的 8000**

`kubectl -n default port-forward svc/pred-svc 8080:8000`

**另开一个终端做健康检查**

`curl -fsS http://127.0.0.1:8080/healthz`

**查看暴露的端口**

`curl -fsS http://127.0.0.1:8080/openapi.json | jq '.paths | keys'`

**转码任务触发**

`curl -fsS -X POST   -F "time_budget_s=20000"   -F "file=@/home/haga/NDSS.mp4;type=video/mp4"   http://127.0.0.1:8080/predict/transcode/upload | jq`

**GEMM算力任务触发**

`curl -s -F time_budget_s=0.1 -F file=@/home/haga/pred-svc/test/gemm/complex_gemm.c   http://127.0.0.1:8080/predict/gemm/from_c_upload | jq .`

**FFT算力任务触发**

`curl -s -F time_budget_s=0.001 -F file=@/home/haga/pred-svc/ldpreload/fft/fft_test.c   http://127.0.0.1:8080/predict/fft/from_c_upload | jq .`
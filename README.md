#  Workshop on time series analytics at JOTB'19.

In this workshop we give a spin on time series analytics. In particular, we avoid the traditional
computation of statistics to focus on patterns, motifs, discords, shapelets and so on.
Many of the algorithms and use cases used in this workshop has been first published by the research team
of Professor **Eamonn Keogh**.

## Getting started

To set-up the environment, yo need to install **Docker** on your machine. After that, you need to perform the following steps:

Go to the directory where you have clone this repository. If you have an NVidia graphics card and want to exploit its capabilities, follow GPU section, if you don't, just go throught the CPU section:

```
* CPU:
  docker build --rm -t shapelets/timeseries-workshop .
  docker run --rm -p 8888:8888 -v <absolute path where you cloned the repo>:/home/khiva-binder -ti shapelets/timeseries-workshop
  open your browser and copy the url that is printed in the docker log, (similar to this one: http://localhost:8888/?token=<jupyter token>)

* GPU:
  docker build --rm -t shapelets/timeseries-workshop . –f Dockerfile-cuda
  docker run –rm --runtime=nvidia -p 8888:8888 -v <absolute path where you cloned the repo>:/home/khiva-binder -ti shapelets/timeseries-workshop
  open your browser and copy the url that is printed in the docker log, (similar to this one: http://localhost:8888/?token=<jupyter token>)
```
## Parts of the workshop

* [MatrixProfile](https://github.com/shapelets/timeseries-workshop/blob/master/matrixprofile/README.md)
* [Distance Metrics](https://github.com/shapelets/timeseries-workshop/blob/master/distances/README.md)
* [Clustering methods](https://github.com/shapelets/timeseries-workshop/blob/master/clustering/README.md)

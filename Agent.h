#ifndef AGENT_H
#define AGENT_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

class Agent {
public:
	int startX, startY;
	int endX, endY;
	int id;

	__device__ __host__ Agent();
	__device__ __host__ Agent(int cId, int cStartX, int cStartY, int cEndX, int cEndY);

	__device__ __host__ void set_id(int cId);
	__device__ __host__ void set_startX(int cStartX);
	__device__ __host__ void set_startY(int cStartY);
	__device__ __host__ void set_endX(int xEndX);
	__device__ __host__ void set_endY(int cEndY);

	__device__ __host__ int get_id(){return id;};
	__device__ __host__ int get_startX(){return startX;};
	__device__ __host__ int get_startY(){return startY;};
	__device__ __host__ int get_endX(){return endX;};
	__device__ __host__ int get_endY(){return endY;};
};

#endif /* AGENT_H */
#include "Agent.h"

//CONSTRUCTORS
__device__ __host__ Agent::Agent(){}

__device__ __host__ Agent::Agent(int cId, int cStartX, int cStartY, int cEndX, int cEndY){
	startX = cStartX;
	startY = cStartY;
	endX = cEndX;
	endY = cEndY;
}

//SETTERS
void Agent::set_id(int cId){
	id = cId;
}

void Agent::set_startX(int cStartX){
	startX = cStartX;
}

void Agent::set_startY(int cStartY){
	startY = cStartY;
}

void Agent::set_endX(int cEndX){
	endX = cEndX;
}

void Agent::set_endY(int cEndY){
	endY = cEndY;
}
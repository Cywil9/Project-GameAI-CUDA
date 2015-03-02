//Jakub Cywinski	C00159241
//4th year Software Development
//Institute of Technology Carlow

//Initially, the Node class and LinkedList class were not included in this file. 
//These classes had to be placed within main file(kernel.cu) in order for the code
//to compile correctly, as there was a linking problem between header files and CUDA
//CUDALINK : nvlink error : Undefined reference to "as123dNodea21sd"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>     /* abs */
#include <ctime>

#define BLOCK_SIZE 16
#define GRID_SIZE 1
const int NO_LISTS = 2;
const int N = 8;
const int W = 0; //WALKABLE
const int S = 1; //START POINT
const int E = 2; //END POINT
const int O = 3; //OBSTRUCTION

//base agent class
class Agent{
public:
	int id;
	int result;

	//constructors
	__device__ __host__ Agent(){}
	__device__ __host__ Agent(int cId, int cResult){
		id = cId;
		result = cResult;
	}

	//setters
	__device__ __host__ void set_id(int cId){ id = cId; }
	__device__ __host__ void set_result(int cResult){ result = cResult; }

	//getters
	__device__ __host__ int get_id(){ return id; }
	__device__ __host__ int get_result() {return result; }
};

//agent class for A* search
class AgentSearch: public Agent{
public:
	int startX, startY;
	int endX, endY;

	__device__ __host__ AgentSearch(){}

	//agent constructor for a* search
	__device__ __host__ AgentSearch(int cId, int cResult, int cStartX, int cStartY, int cEndX, int cEndY){
		id = cId;
		result = cResult;
		startX = cStartX;
		startY = cStartY;
		endX = cEndX;
		endY = cEndY;
	}

	//setters
	__device__ __host__ void set_startX(int cStartX){ startX = cStartX; }
	__device__ __host__ void set_startY(int cStartY){ startY = cStartY; }
	__device__ __host__ void set_endX(int cEndX){ endX = cEndX; }
	__device__ __host__ void set_endY(int cEndY){ endY = cEndY; }

	//getters
	__device__ __host__ int get_startX(){ return startX; }
	__device__ __host__ int get_startY(){ return startY; }
	__device__ __host__ int get_endX(){ return endX; }
	__device__ __host__ int get_endY(){ return endY; }
};

//agent class for finite state machines
class AgentFSM: public Agent{
	int fuellvl;
	int distToCov;

	//agent constructor for fsm
	__device__ __host__ AgentFSM(int cId, int cResult, int cFuellvl, int cDistToCov){
		id = cId;
		result = cResult;
		fuellvl = cFuellvl;
		distToCov = cDistToCov;
	}

	//setters
	__device__ __host__ void set_fuellvl(int cFuellvl){ fuellvl = cFuellvl; }
	__device__ __host__ void set_distToCov(int cDistToCov){ distToCov = cDistToCov; }

	//getters
	__device__ __host__ int get_fuellvl(){ return fuellvl; }
	__device__ __host__ int get_distToCov(){ return distToCov; }
};

//agent class for decision trees
class AgentDT: public Agent{
	//decisions will be randomised
	//d1 -> first level
	//d2 -> second level
	//etc.
	int d1, d2, d3, d4, d5, d6;

	//agent constructor for dt -> 3 levels(minimum, at the moment anyway)
	__device__ __host__ AgentDT(int cId, int cResult, int cD1, int cD2, int cD3){
		id = cId;
		result = cResult;
		d1 = cD1;
		d2 = cD2;
		d3 = cD3;
	}

	//4 levels
	__device__ __host__ AgentDT(int cId, int cResult, int cD1, int cD2, int cD3, int cD4){
		id = cId;
		result = cResult;
		d1 = cD1;
		d2 = cD2;
		d3 = cD3;
		d4 = cD4;
	}

	//5 levels
	__device__ __host__ AgentDT(int cId, int cResult, int cD1, int cD2, int cD3, int cD4, int cD5){
		id = cId;
		result = cResult;
		d1 = cD1;
		d2 = cD2;
		d3 = cD3;
		d4 = cD4;
		d5 = cD5;
	}

	//6 levels
	__device__ __host__ AgentDT(int cId, int cResult, int cD1, int cD2, int cD3, int cD4, int cD5, int cD6){
		id = cId;
		result = cResult;
		d1 = cD1;
		d2 = cD2;
		d3 = cD3;
		d4 = cD4;
		d5 = cD5;
		d6 = cD6;
	}

	//setters
	__device__ __host__ void set_d1(int cD1){ d1 = cD1; }
	__device__ __host__ void set_d2(int cD2){ d2 = cD2; }
	__device__ __host__ void set_d3(int cD3){ d3 = cD3; }
	__device__ __host__ void set_d4(int cD4){ d4 = cD4; }
	__device__ __host__ void set_d5(int cD5){ d5 = cD5; }
	__device__ __host__ void set_d6(int cD6){ d6 = cD6; }

	//getters
	__device__ __host__ int get_d1(){ return d1; }
	__device__ __host__ int get_d2(){ return d2; }
	__device__ __host__ int get_d3(){ return d3; }
	__device__ __host__ int get_d4(){ return d4; }
	__device__ __host__ int get_d5(){ return d5; }
	__device__ __host__ int get_d6(){ return d6; }

};


class Node{
public:
	int status;
	int row;
	int col;
	int costG;		//move cost from starting point
	int costH;		//estimated move cost to final destination (10 per square)
	int totalF;		// G + H
	int agentId;
	Node *parent; //pointer to parent node

	//Node constructor
	__device__ __host__ Node(){}

	__device__ __host__ Node(int nStatus){
		status = nStatus;
	}

	__device__ __host__ Node(int nStatus, int nRow, int nCol){
		status = nStatus;
		row = nRow;
		col = nCol;
	}

	__device__ __host__ Node(int nStatus, int nRow, int nCol, int nCostG, int nCostH, int nTotalF, Node *nParentId){
		status = nStatus;
		row = nRow;
		col = nCol;
		costG = nCostG;	
		costH = nCostH;	
		totalF = nTotalF;
		parent = parent;
	}

	//SETTERS
	__device__ __host__ void set_status(int nStatus){
		status = nStatus;
	}
	__device__ __host__ void set_row(int nRow){
		row = nRow;
	}
	__device__ __host__ void set_col(int nCol){
		col = nCol;
	}
	__device__ __host__ void set_costG(int nCostG){
		costG = nCostG;
	}
	__device__ __host__ void set_costH(int nCostH){
		costH = nCostH;
	}
	__device__ __host__ void set_totalF(int nTotalF){
		totalF = nTotalF;
	}

	//REVIEW
	__device__ __host__ void set_parent(Node *nParent){
		parent = nParent;
	}

	__device__ __host__ void set_agentId(int nAgentId){
		agentId = nAgentId;
	}

	__device__ __host__ int get_status() { return status; }
	__device__ __host__ int get_row() { return row; }
	__device__ __host__ int get_col() { return col; }
	__device__ __host__ int get_costG() { return costG; }
	__device__ __host__ int get_costH() { return costH; }
	__device__ __host__ int get_totalF() { return totalF; }
	__device__ __host__ int get_agentId() { return agentId; }
	__device__ __host__ Node get_parent() { return *parent; }
};

class LinkedList{
	__device__ __host__ struct Element{
		Node n1;
		Element *next;
	};

public:
	__device__ __host__ LinkedList(){
		head = NULL;
	}

	__device__ __host__ void addNode(Node funcN){
		Element *el = new Element();
		el->n1.set_status(funcN.get_status());
		el->n1.set_row(funcN.get_row());
		el->n1.set_col(funcN.get_col());
		el->n1.set_costG(funcN.get_costG());
		el->n1.set_costH(funcN.get_costH()); 
		el->n1.set_totalF(funcN.get_totalF());
//		*el->n1.set_parent(funcN.get_parent());

		el->next = head;
		head = el;
	}

	__device__ __host__ bool isEmpty(){
		Element *cur = head;
		
		if(cur == NULL){
			return true;
		}

		return false;
	}

	//check if node already in the list
	__device__ __host__ bool contains(int x, int y){
		Element *cur = head;

		while(cur != NULL){
			if((cur->n1.get_row() == x) && (cur->n1.get_col() == y)){
			//	if(cur->n1.get_totalF() < funcN.get_totalF()){
					return true;
			//	}
			}
			cur = cur->next;
		}

		return false;
	}

	__device__ __host__ bool update(int x, int y, int inG, Node inQ){
		Element *cur = head;
		int newF = 0;
		while(cur != NULL){
			if((cur->n1.get_row() == x) && (cur->n1.get_col() == y)){
				newF = inG + cur->n1.get_costH();
				if(cur->n1.get_totalF() > newF){
					cur->n1.set_costG(inG);
					cur->n1.set_totalF(newF);
					cur->n1.set_parent(&inQ);
					return true;
				}
			}
			cur = cur->next;
		}

		return false;
	}
	
	__device__ __host__ Node findLowestF(){
		int index =0;
		Element *cur = head;

		Node lowest;

		while(cur != NULL){
			if(index == 0){
				lowest.set_status(cur->n1.get_status());
				lowest.set_row(cur->n1.get_row());
				lowest.set_col(cur->n1.get_col());
				lowest.set_costG(cur->n1.get_costG());
				lowest.set_costH(cur->n1.get_costH());
				lowest.set_totalF(cur->n1.get_totalF());
			}
			else if(cur->n1.get_totalF() < lowest.get_totalF()){
				lowest.set_status(cur->n1.get_status());
				lowest.set_row(cur->n1.get_row());
				lowest.set_col(cur->n1.get_col());
				lowest.set_costG(cur->n1.get_costG());
				lowest.set_costH(cur->n1.get_costH());
				lowest.set_totalF(cur->n1.get_totalF());
//				lowest->n1.set_parent(cur->n1.get_parent());
			}
			cur = cur->next;
			index++;
		}
		return lowest;
	}

	//does not work!!!
	__device__ __host__ Node popFirstNode(){
		Element *cur = head;
		Node n;

		if(cur != NULL){
			n = cur -> n1;
			head = head -> next;
		}
		delete cur;
		return n;
	}

	__device__ __host__ int removeNode(Node popNode){
		Element *cur = head, *prev;

		while(cur != NULL){
			if((cur->n1.get_totalF() == popNode.get_totalF() )&& (cur->n1.get_row() == popNode.get_row()) && (cur->n1.get_col() == popNode.get_col())) {
				//&& (cur->n1.get_row() == popNode.get_row()) && (cur->n1.get_col() == popNode.get_col())){
				if(cur == head){
					head = cur->next;
					delete cur;
					return 1;
				}
				else{
					prev->next = cur->next;
					delete cur;
					return 1;
				}
			}
			else{
				prev = cur;
				cur = cur->next;
			}
		}
		return 0;
	}
private:
	Element *head;
};
/*
class State {
private:
	int this_state;
public:
	__device__ __host__ State(int new_state) : this_state(new_state) { };
	__device__ __host__ virtual State * change_state(int choice) = 0;
	__device__ __host__ void wait_a_second();
};

//locked state
class StartState : public State {
public:
	__device__ __host__ StartState() : State(1) { };
	__device__ __host__ State * change_state(int choice);
};

//unlock state
class DrivingState : public State {
private:
public:
	int fuellvl;
	int distcov;
	__device__ __host__ DrivingState() : State(2), fuellvl(5), distcov(0) { };
	__device__ __host__ State * change_state(int choice);
};

class GetFuelState : public State {
public:
	__device__ __host__ GetFuelState() : State(3) { };
	__device__ __host__ State * change_state(int choice);
};

class EndState : public State {
public:
	__device__ __host__ EndState() : State(3) { };
	__device__ __host__ State * change_state(int choice);
};

__device__ __host__ void State::wait_a_second() {
    time_t time_current = time(NULL);
	while ( time(NULL) <= time_current+1 );
}

//locked state. transition to unlocked state
__device__ __host__ State * StartState::change_state(int choice) {
	std::cout << "Start" << std::endl;
	if(choice == 0)
		reinterpret_cast<DrivingState *>(this)->DrivingState::DrivingState();
	else
		reinterpret_cast<DrivingState *>(this)->DrivingState::DrivingState();
	wait_a_second();
	return this;
}

__device__ __host__ State * DrivingState::change_state(int choice) {
	std::cout << "Driving " << std::endl;
	distcov++;

	if(choice == 1)
		reinterpret_cast<GetFuelState *>(this)->GetFuelState::GetFuelState();
	else if(choice == 2){
		reinterpret_cast<DrivingState *>(this)->DrivingState::DrivingState();
	}
	else if(distcov > 2)
		reinterpret_cast<EndState *>(this)->EndState::EndState();

	wait_a_second();
	return this;
}

__device__ __host__ State * GetFuelState::change_state(int choice) {
	std::cout << "Getting Fuel" << std::endl;
	reinterpret_cast<DrivingState *>(this)->DrivingState::DrivingState();
	wait_a_second();
	return this;
}

__device__ __host__ State * EndState::change_state(int choice) {
	std::cout << "The end" << std::endl;
	reinterpret_cast<StartState *>(this)->StartState::StartState();
	wait_a_second();
	return this;
};

class CarFSM {
private:
	State * current_state;
public:
	__device__ __host__ CarFSM() : current_state(new StartState()) { srand(static_cast<unsigned int>(time(NULL))); };
	__device__ __host__ void next_state(int choice) { current_state = current_state->change_state(choice); };
	__device__ __host__ void simulate_car() {
		while( true ){
			int state_shift = rand()%2;
			next_state(state_shift);
		}
	};
};
*/

typedef enum{
	StartState, DrivingState, GetFuelState, EndState
} my_state_t;

//CPU execution
void carFSM(int tankSize, int distToCover){
	my_state_t state = StartState;

	int fuellvl = tankSize;	//full tank at the start
	int distCov = 0;
	bool finished = false;
	while(!finished){
		switch(state){
			//START STATE
			case StartState:
				std::cout << "Start State" << std::endl;
				state = DrivingState;
				break;
			//DRIVING STATE
			case DrivingState:
				//check if the distance covered so far is not the goal distance
				if(distCov != distToCover){
					//check if the car needs to get fuel
					if(fuellvl < 2){
						state = GetFuelState;
					}
					//proceed if theres more distance to cover & car has fuel
					else{
						std::cout << "Driving State" << std::endl;
						state = DrivingState;
						distCov++;	//distance + 1
						fuellvl--;	//fuel - 1
					}
				}
				//when at goal
				else
					state = EndState;
				break;
			//getting fuel
			case GetFuelState:
				std::cout << "Get Fuel State" << std::endl;
				//fuel up by 5
				fuellvl+= 5;
				state = DrivingState;
				break;
			//finished, at destination
			case EndState:
				std::cout << "End State" << std::endl;
				//end while loop 
				finished = true;
				break;
		}
	}
}

//CUDA execution
//int tankSize, int distToCover <-- randomise? 
__global__ void carFSM(float* d_array, float* destinationArray, size_t pitch, int rows, int cols){
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int fuellvl = 5;		//full tank at the start
	int distToCover = 10;	//distance to be covered by all agents <-- randomise?
	int distCov = 0;		//distance covered so fat
	int allStates[30];		//30 -> max number of agents, stores current state of agent
	const int startState = 1;
	const int driveState = 2;
	const int getFuelState = 3;
	const int endState = 4;

	if(idx < rows){
		bool finished = false;
		int index = 0;
		allStates[idx] = startState; //store current state
		index++;

		while(!finished){
			switch(allStates[idx]){
				//START STATE
				case startState:
				//	std::cout << "Start State" << std::endl;
					allStates[idx] = driveState;
					index++;
					break;
				//DRIVING STATE
				case driveState:
					//check if the distance covered so far is not the goal distance
					if(distCov != distToCover){
						//check if the car needs to get fuel
						if(fuellvl < 2){
							allStates[idx] = getFuelState;
							index++;
						}
						//proceed if theres more distance to cover & car has fuel
						else{
						//	std::cout << "Driving State" << std::endl;
							allStates[idx] = driveState;
							distCov++;	//distance + 1
							fuellvl--;	//fuel - 1
							index++;
						}
					}
					//when at goal
					else{
						allStates[idx] = endState;
						index++;
					}
					break;
				//GETTING FUEL
				case getFuelState:
				//	std::cout << "Get Fuel State" << std::endl;
					//fuel up by 5
					fuellvl+= 5;
					allStates[idx] = driveState;
					index++;
					break;
				//END STATE --> at destination
				case endState:
				//	std::cout << "End State" << std::endl;
					allStates[idx] = endState;
					//end while loop 
					finished = true;
					index++;
					break;
			}
		}
		destinationArray[idx] = index;
	}
}

//CPU execution
void carDT(int agents){
	int i = 0;
	srand( time(NULL) ); 
	int atDest = rand()%2;
	int parkAv = rand()%2;
	int lowPetrol = rand()%2;
		//if at destination
		if(atDest == 1){
			//if parking available
			if(parkAv == 1){
				std::cout << "You are at destination. Car Parked." << std::endl;
			}
			//if parking unavailable
			else{
				std::cout << "You are at destination. Looking for parking." << std::endl;
			}
		}
		//if not at destination
		else{
			//if low on petrol
			if(lowPetrol == 1){
				std::cout << "Not at destination. Low petrol - looking for a petrol station." << std::endl;
			}
			//if not low on petrol
			else{
				std::cout << "Not at destination. Still driving." << std::endl;
			}
		}
	
}
/*
//CUDA execution
__global__ void carDT(int agents, int asd){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	srand( time(NULL) ); 
	int atDest = rand()%2;
	int parkAv = rand()%2;
	int lowPetrol = rand()%2;
	if(i < agents){
		//if at destination
		if(atDest == 1){
			//if parking available
			if(parkAv == 1){
		//		std::cout << "You are at destination. Car Parked." << std::endl;
			}
			//if parking unavailable
			else{
		//		std::cout << "You are at destination. Looking for parking." << std::endl;
			}
		}
		//if not at destination
		else{
			//if low on petrol
			if(lowPetrol == 1){
			//	std::cout << "Not at destination. Low petrol - looking for a petrol station." << std::endl;
			}
			//if not low on petrol
			else{
			//	std::cout << "Not at destination. Still driving." << std::endl;
			}
		}
	}
}
*/

//!!---CODE FROM http://choorucode.com/2011/04/09/thrust-passing-device_vector-to-kernel/ ---!!
// Template structure to pass to kernel
template < typename T >
struct KernelArray
{
    T*  _array;
    int _size;
};
 
// Function to convert device_vector to structure
template < typename T >
KernelArray< T > convertToKernel( thrust::device_vector< T >& dVec )
{
    KernelArray< T > kArray;
    kArray._array = thrust::raw_pointer_cast( &dVec[0] );
    kArray._size  = ( int ) dVec.size();
 
    return kArray;
}
//!!---CODE FROM http://choorucode.com/2011/04/09/thrust-passing-device_vector-to-kernel/ ENDS HERE---!!
//------------------------------------------------------------------------------------------------------

//A* Search
__global__ void myKernel(Node* d_allNodesArr, Node* destinationArr, size_t pitch, int cols, int rows, KernelArray<Node> DevStartNodeArray, KernelArray<Node> DevEndNodeArray, int nodeArrLen){
	
	int idx=blockIdx.x*blockDim.x + threadIdx.x;
	//number of squares from start node
	int G = 0;
	//number of squares from end node
	int	H = 0;
	//total cost
	int F;
	int parX, parY;				//parent node
	int curX, curY;				//current node
	int d_startX, d_startY;		//start node
	int d_endX, d_endY;			//end node
	bool check = false;
	//init open list
	LinkedList openList[NO_LISTS];
	//init closed list
	LinkedList closedList[NO_LISTS];

	int mainIndex = 0;
	bool first = false;
	bool second = false;

	if(idx < nodeArrLen){
		d_startX = DevStartNodeArray._array[idx].get_row();
		d_startY = DevStartNodeArray._array[idx].get_col();

		d_endX = DevEndNodeArray._array[idx].get_row();
		d_endY = DevEndNodeArray._array[idx].get_col();
		Node endNode;
		endNode.set_row(d_endX);
		endNode.set_col(d_endY);

		//add start node to openList
		//used the link below to fix ptxas fatal error
		//http://stackoverflow.com/questions/17188527/cuda-external-class-linkage-and-unresolved-extern-function-in-ptxas-file
		Node n;
		n.set_status(S);
		n.set_row(d_startX);
		n.set_col(d_startY);
		n.set_totalF(0);
		//put the starting node in open list
		openList[idx].addNode(n);


	//	Node n(W, d_startX, d_startY); //<--------DOESNT WORK -- ptxas fatal   : Unresolved extern function
	//	openList[idx].addNode(d_startX, d_startY, 0, 0, 0, 0, 0);


	/*	DO NOT DELETE THIS
		--------------------------------
		Node n1;
		n1.set_status(S);
		n1.set_row(3);
		n1.set_col(3);
		n1.set_totalF(30);
		//put the starting node in open list
		openList[idx].addNode(n);
		openList[idx].addNode(n1);

		Node newlowest = openList[idx].findLowestF();
		destinationArr[0] = newlowest;
		openList[idx].removeNode(newlowest);

		Node onemore = openList[idx].findLowestF();
		destinationArr[1] = onemore;
		--------------------------------- */	

		//loop until open list has elements
		while(openList[idx].isEmpty() == false){
			Node q = openList[idx].findLowestF();
			closedList[idx].addNode(q);
			openList[idx].removeNode(q);

			//check if end node is in closed list
			if(closedList[idx].contains(endNode.get_row(), endNode.get_col())){
				//path found
				break;
			}

			curX = q.get_row();
			curY = q.get_col();
			

			for(int i = curX - 1; i <= curX + 1; ++i){
				Node* rowData = (Node*)((char*)d_allNodesArr + i * pitch);  
				for (int j = curY - 1; j <= curY + 1; ++j) {
					if(((i == curX) == (j == curY))) continue;
		//			if(i == curX && j == curY) continue;
					//if adjacent node is already in the closed list skip it
					if(closedList[idx].contains(i,j)) continue;
				
					int checkI =i;
					int checkJ =j;
					
					G = std::abs(i - d_startX) + std::abs(j - d_startY);

					//if its not in the open list
					if(!openList[idx].contains(i,j)){
						//COMPUTE ITS SCORE AND SET THE PARENT
						Node adjNode;
						adjNode.set_row(checkI);
						adjNode.set_col(checkJ);

						//G
						adjNode.set_costG(G);
						//H
						H = std::abs(i - d_endX) + std::abs(j - d_endY);
						adjNode.set_costH(H);		
						//F
						F = G + H;
						adjNode.set_totalF(F);
						//set parent to q
						adjNode.set_parent(&q);
						//add the new adjacent node to open list
						openList[idx].addNode(adjNode);
					}
					else{
						openList[idx].update(i, j, G, q);
					}
				}
			}
		}
		int thisIndex = 0;
		while(!closedList[idx].isEmpty()){
			Node first = closedList[idx].popFirstNode();
			destinationArr[thisIndex] = first;
			thisIndex++;
		}
	}
}


int main(){
	int rows, cols;
	int noAgents;
	int choice;

	std::cout << "Choose one" << std::endl;
	std::cout << "1: A* Search" << std::endl;
	std::cout << "2: Finite State Machine" << std::endl;
	std::cout << "3: Decision Trees" << std::endl;
	std::cin >> choice;


	//A* SEARCH
	if(choice == 1){
		rows = N, 
		cols = N;
		int vectLen = 0;
		int startX, startY;
		int endX, endY;

		//agent list
		std::vector<AgentSearch> agentList;
		AgentSearch a1;

		//start node list
		std::vector<Node> startNodeList;
		Node n1;

		//end node list
		std::vector<Node> endNodeList;
		Node n2;

		//grid array
		Node allNodesArr[N][N];	//8x8
		Node* d_allNodesArr; // the device array which memory will be allocated to
		Node* d_destinationArr;

		//populate all nodes, default status
		for(int i = 0; i < N; i++){
			for(int j = 0; j < N; j++){
				allNodesArr[i][j].set_status(W);
				allNodesArr[i][j].set_row(i);
				allNodesArr[i][j].set_col(j);
			}
		}

		//allocate memory on the host
		Node* h_array = new Node[rows * cols];
		//pitch value is assigned by cudaMallocPitch, provides correct data structure alignment
		size_t pitch;
		//memory for source array
		cudaMallocPitch(&d_allNodesArr, &pitch, cols * sizeof(Node), rows);
		//memory for dest array
		cudaMalloc(&d_destinationArr, cols*rows*sizeof(Node));

		//user input
		std::cout << "Please enter the number of Agents: " << std::endl;
		std::cin >> noAgents;
	
		for(int i = 0; i < noAgents; i++){
			std::cout << "Agent " << i+1 << " Please enter START X: " << std::endl;
			std::cin >> startX;

			std::cout << "Agent " << i+1 << " Please enter START Y: " << std::endl;
			std::cin >> startY;

			std::cout << "Agent " << i+1 << " Please enter END X: " << std::endl;
			std::cin >> endX;

			std::cout << "Agent " << i+1 << " Please enter END Y: " << std::endl;
			std::cin >> endY;

			a1.set_id(i);
			a1.set_startX(startX);
			a1.set_startY(startY);
			a1.set_endX(endX);
			a1.set_endY(endY);
			agentList.push_back(a1);
		}

		//iterate through the agentList
		//and populate start node and end node vectors
		std::vector<AgentSearch>::iterator agentIt;
		for(agentIt = agentList.begin(); agentIt != agentList.end(); ++agentIt){
			//start node
			n1.set_status(S);
			n1.set_row(agentIt->get_startX());
			n1.set_col(agentIt->get_startY());
			n1.set_agentId(agentIt->get_id());
			startNodeList.push_back(n1);
		
			//end node
			n2.set_status(E);
			n2.set_row(agentIt->get_endX());
			n2.set_col(agentIt->get_endY());
			n2.set_agentId(agentIt->get_id());
			endNodeList.push_back(n2);
		}
		
		//convert the nodeList vector to thrust vector
		thrust::device_vector<Node> DevStartNodeList(startNodeList.begin(), startNodeList.end());
		thrust::device_vector<Node> DevEndNodeList(endNodeList.begin(), endNodeList.end());
		//vector length
		vectLen = startNodeList.end() - startNodeList.begin();

		myKernel<<< 100, 512 >>> (d_allNodesArr, d_destinationArr, pitch, cols, rows, convertToKernel(DevStartNodeList), convertToKernel(DevEndNodeList), vectLen);

		//return
		cudaMemcpy(h_array, d_destinationArr, cols*rows*sizeof(Node), cudaMemcpyDeviceToHost);  
	
		for(int i = 0 ; i < rows; i++)  
		  {  
			for(int j = 0 ; j < cols ; j++)  
			{  
				std::cout << "h_array[" << (i*cols) + j << "]=" << h_array[(i*cols) + j].get_row() << ", " << h_array[(i*cols) + j].get_col() << std::endl;  
			}  
		}  
	}
	//FINITE STATE MACHINE
	else if(choice == 2){
		//ask user if they want to execute FSM on the cpu or cuda
		int c = 0;
		std::cout << "1: CPU execution" << std::endl;
		std::cout << "2: CUDA execution" << std::endl;
		std::cin >> c;

		//cpu execution
		if(c == 1){
			carFSM(7, 10);
		}
		//cuda execution
		else if(c == 2){
			//agents = number of rows on cuda
			std::cout << "Enter the number of agents" << std::endl;
			std::cin >> noAgents;
			rows = noAgents;
			//states = columns
			cols = 30;	//max number of states per agent

			float* d_array; //device array which memory will be allocated to
			float* d_destinationArray; //device array

			//allocate memory on the host
			float* h_array = new float[rows*cols];

			//the pitch values is assigned by cudaMallocPitch 
			//it ensures correct data structure alignment
			size_t pitch;

			//allocated the device memory for source array
			cudaMallocPitch(&d_array, &pitch, rows*sizeof(float), cols);

			//allocate the device memory for destination array
			cudaMalloc(&d_destinationArray, rows*cols*sizeof(float));

			//call the kernel which copies values for d_array to d_destinationArray
			carFSM<<<100, 512>>>(d_array, d_destinationArray, pitch, rows, cols);

			//copy the data back to the host memory
			cudaMemcpy(h_array, d_destinationArray, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

			for(int i = 0; i < rows; i++){
				for(int j = 0; j < cols; j++){
					std::cout << "h_array[" << (i*cols) +j << "]="<< h_array[(i*cols) + j] << std::endl;
				}
			}
		}
		else
			std::cout << "ERROR, wrong number" << std::endl;
	}
	//DECISION TREES
	else if(choice == 3){
		//ask user if they want to execute decision trees on the cpu or cuda
		int c = 0;
		std::cout << "1: CPU execution" << std::endl;
		std::cout << "2: CUDA execution" << std::endl;
		std::cin >> c;

		//cpu execution
		if(c == 1){
			std::cout << "Enter the number of agents" << std::endl;
			std::cin >> noAgents;
			carDT(noAgents);
		}
		//cuda execution
		else if(c == 2){
			std::cout << "Enter the number of agents" << std::endl;
			std::cin >> noAgents;

		//	carDT<<<256, 256>>>(noAgents);
		}
		else
			std::cout << "ERROR, wrong number" << std::endl;

	}
	else{
		std::cout << "ERROR, wrong number" << std::endl;
	}
}
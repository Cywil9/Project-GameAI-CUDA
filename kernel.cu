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

class Node{
public:
	int id;
	int status;
	int row;
	int col;
	int costG;		//move cost from starting point
	int costH;		//estimated move cost to final destination (10 per square)
	int totalF;		// G + H
	Node *parent; //pointer to parent node

	//Node constructor
	__device__ __host__ Node(){}

	__device__ __host__ Node(int cId, int cStatus){
		id = cId;
		status = cStatus;
	}

	__device__ __host__ Node(int cId, int cStatus, int cRow, int cCol){
		id = cId;
		status = cStatus;
		row = cRow;
		col = cCol;
	}

	__device__ __host__ Node(int cId, int cStatus, int cRow, int cCol, int cCostG, int cCostH, int cTotalF, Node *cParent){
		id = cId;
		status = cStatus;
		row = cRow;
		col = cCol;
		costG = cCostG;	
		costH = cCostH;	
		totalF = cTotalF;
		parent = cParent;
	}

	//SETTERS
	__device__ __host__ void set_id(int cId){
		id = cId;
	}
	__device__ __host__ void set_status(int cStatus){
		status = cStatus;
	}
	__device__ __host__ void set_row(int cRow){
		row = cRow;
	}
	__device__ __host__ void set_col(int cCol){
		col = cCol;
	}
	__device__ __host__ void set_costG(int cCostG){
		costG = cCostG;
	}
	__device__ __host__ void set_costH(int cCostH){
		costH = cCostH;
	}
	__device__ __host__ void set_totalF(int cTotalF){
		totalF = cTotalF;
	}

	//REVIEW
	__device__ __host__ void set_parent(Node *cParent){
		parent = cParent;
	}

	__device__ __host__ int get_id() { return id; }
	__device__ __host__ int get_status() { return status; }
	__device__ __host__ int get_row() { return row; }
	__device__ __host__ int get_col() { return col; }
	__device__ __host__ int get_costG() { return costG; }
	__device__ __host__ int get_costH() { return costH; }
	__device__ __host__ int get_totalF() { return totalF; }
	__device__ __host__ Node* get_parent() { return parent; }
};

__device__ __host__ struct Element{
		Node n1;
		Element *next;
};

class LinkedList{
public:
	Element *head;

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
		el->n1.set_parent(funcN.get_parent());

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

	//check if node already in the list NODE
	__device__ __host__ bool contains(Node node){
		Element *cur = head;

		while(cur != NULL){
			if((cur->n1.get_row() == node.get_row()) && (cur->n1.get_col() == node.get_col())){
			//	if(cur->n1.get_totalF() < funcN.get_totalF()){
				return true;
			//	}
			}
			cur = cur->next;
		}

		return false;
	}

		//check if node already in the list X Y
	__device__ __host__ bool contains(int x, int y, int f){
		Element *cur = head;
		
		while(cur != NULL){
			if(((cur->n1.get_row() == x) && (cur->n1.get_col() == y)) && (cur->n1.get_totalF() < f)){
				return true;
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
				//lowest->n1.set_parent(cur->n1.get_parent());
			}
			cur = cur->next;
			index++;
		}
		return lowest;
	}

	//does not work
	//__device__ 
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

};

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
	Node start;
	Node end;
	LinkedList path;

	__device__ __host__ AgentSearch(){}

	//agent constructor for a* search
	__device__ __host__ AgentSearch(int cId, int cResult, Node cStart, Node cEnd, LinkedList cPath){
		id = cId;
		result = cResult;
		start = cStart;
		end = cEnd;
		path = cPath;
	}

	//setters
	__device__ __host__ void set_start(Node cStart){ start = cStart; }
	__device__ __host__ void set_end(Node cEnd){ end = cEnd; }
	__device__ __host__ void set_path(LinkedList cPath){ path = cPath; }

	//getters
	__device__ __host__ Node get_start(){ return start; }
	__device__ __host__ Node get_end(){ return end; }
	__device__ __host__ LinkedList get_path(){ return path; }
};

//agent class for finite state machines
class AgentFSM: public Agent{
public:
	int fuellvl;
	int distToCov;
	int distCovd;
	int currentState;

	//agent constructor for fsm
	__device__ __host__ AgentFSM(){}

	__device__ __host__ AgentFSM(int cId, int cResult, int cFuellvl, int cDistToCov, int cDistCovd, int cCurrentState){
		id = cId;
		result = cResult;
		fuellvl = cFuellvl;
		distToCov = cDistToCov;
		distCovd = cDistCovd;
		currentState = cCurrentState;
	}

	//setters
	__device__ __host__ void set_fuellvl(int cFuellvl){ fuellvl = cFuellvl; }
	__device__ __host__ void set_distToCov(int cDistToCov){ distToCov = cDistToCov; }
	__device__ __host__ void set_distCovd(int cDistCovd){ distCovd = cDistCovd; }
	__device__ __host__ void set_currentState(int cCurrentState){ currentState = cCurrentState; }

	//getters
	__device__ __host__ int get_fuellvl(){ return fuellvl; }
	__device__ __host__ int get_distToCov(){ return distToCov; }
	__device__ __host__ int get_distCovd(){ return distCovd; }
	__device__ __host__ int get_currentState(){ return currentState; }
};

//agent class for decision trees
class AgentDT: public Agent{
public:
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
__global__ void carFSM(AgentFSM* d_array,  int noOfAgents){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int myDistCovd;
	int myFuelLvl;
	const int startState = 1;
	const int driveState = 2;
	const int getFuelState = 3;
	const int endState = 4;
	
	if(idx < noOfAgents){
		bool finished = false;
		int index = 0;
		d_array[idx].set_currentState(startState); //store current state
		index++;

		while(index < 30){
			switch(d_array[idx].get_currentState()){
				//START STATE
				case startState:
				//	std::cout << "Start State" << std::endl;
					d_array[idx].set_currentState(driveState);
					index++;
					break;
				//DRIVING STATE
				case driveState:
					//check if the distance covered so far is not the goal distance
					if(d_array[idx].get_distCovd() != d_array[idx].get_distToCov()){
						//check if the car needs to get fuel
						if(d_array[idx].get_fuellvl() < 2){
							d_array[idx].set_currentState(getFuelState);
							index++;
						}
						//proceed if theres more distance to cover & car has fuel
						else{
						//	std::cout << "Driving State" << std::endl;
							d_array[idx].set_currentState(driveState);
							//update distance covered
							myDistCovd = d_array[idx].get_currentState();
							myDistCovd++;	//distance + 1
							d_array[idx].set_distCovd(myDistCovd);
							//update fuel level
							myFuelLvl = d_array[idx].get_fuellvl();
							myFuelLvl--;	//fuel - 1
							d_array[idx].set_fuellvl(myFuelLvl);
							index++;
						}
					}
					//when at goal
					else{
						d_array[idx].set_currentState(endState);
						index++;
					}
					break;
				//GETTING FUEL
				case getFuelState:
				//	std::cout << "Get Fuel State" << std::endl;
					//fuel up by 1
					myFuelLvl = d_array[idx].get_fuellvl();
					myFuelLvl += 1;
					d_array[idx].set_fuellvl(myFuelLvl);
					d_array[idx].set_currentState(driveState);
					index++;
					break;
				//END STATE --> at destination
				case endState:
				//	std::cout << "End State" << std::endl;
					d_array[idx].set_currentState(endState);
					//end while loop 
					finished = true;
					index++;
					break;
			}
		}
		//number of states
		d_array[idx].set_result(index);
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

//CUDA execution
__global__ void carDT(AgentDT* d_array,  int noOfAgents){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
/*	srand( time(NULL) ); 
	int atDest = rand()%2;
	int parkAv = rand()%2;
	int lowPetrol = rand()%2;*/
	/* 1 - You are at destination. Car Parked
	*  2 - You are at destination. Looking for parking
	*  3 - Not at destination. Low petrol - looking for a petrol station
	*  4 - Not at destination. Still driving
	*/
	if(idx < noOfAgents){
		//if at destination
		if(d_array[idx].get_d1() == 1){
			//if parking available
			if(d_array[idx].get_d2() == 1){
		//		std::cout << "You are at destination. Car Parked." << std::endl;
				d_array[idx].set_result(1);
			}
			//if parking unavailable
			else{
		//		std::cout << "You are at destination. Looking for parking." << std::endl;
				d_array[idx].set_result(2);
			}
		}
		//if not at destination
		else{
			//if low on petrol
			if(d_array[idx].get_d2() == 1){
			//	std::cout << "Not at destination. Low petrol - looking for a petrol station." << std::endl;
				d_array[idx].set_result(3);
			}
			//if not low on petrol
			else{
			//	std::cout << "Not at destination. Still driving." << std::endl;
				d_array[idx].set_result(4);
			}
		}
	}
}

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

//A* on cpu
void AStart(Node* allNodes){
	int curX, curY;
	int G, H, F;


}

//A* Search
__global__ void searchKernel(AgentSearch* d_Agents, Node* d_AllNodes, const int noOfAgents){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int curX, curY;
	int G, H, F;
	//array of successors
	//init open list	
	LinkedList openList[10];
	//init closed list
	LinkedList closedList[10];
	bool found = false;

	if(idx < noOfAgents){
		Node successors[3];
		Node n = d_Agents[idx].get_start();
		n.set_status(S);
		n.set_totalF(0);

		openList[idx].addNode(n);

		while(openList[idx].isEmpty() == false){
			//find the node with the least f on the open list, call it "q
			Node q = openList[idx].findLowestF();
			//pop q off the open list
			openList[idx].removeNode(q);

			if(closedList[idx].contains(d_Agents[idx].get_end())){
				//path found
				found = true;
				break;
			}

			curX = q.get_row();
			curY = q.get_col();

			int k=0;
			//generate q's 84successors and set their parents to q
			for(int i = curX - 1; i <= curX + 1; ++i){
				for(int j = curY - 1; j <= curY + 1; ++j){
					//d_AllNodes[i * 8 * j]; 8= width
					if((i == curX) == (j == curY)) continue;
					else{
						Node s(0, W, i, j);
						s.set_parent(&q);
						successors[k] = s;
						k++;
					}
				}
			}
			//for earch successor
			for(int i = 0; i <= 3; ++i){
				//get successors X & Y
				int scsrX = successors[i].get_row();
				int scsrY = successors[i].get_col();

				//if successor is the goal, stop the search
				if((scsrX == d_Agents[idx].get_end().get_row()) && (scsrY == d_Agents[idx].get_end().get_col())){
					//stop search
					found = true;
					break;
				}
				else{
					//successor.g = q.g + distance between successor and q
					successors[i].set_costG(q.get_costG() +(std::abs(curX - d_Agents[idx].get_start().get_row()) + std::abs(curY - d_Agents[idx].get_start().get_col())));
					//successor.h = distance from goal to successor
					successors[i].set_costH(std::abs(curX - d_Agents[idx].get_end().get_row()) + std::abs(curY - d_Agents[idx].get_end().get_col()));
					//successor.f = successor.g + successor.h
					successors[i].set_totalF(successors[i].get_costG() + successors[i].get_costH());

					//if a node with the same position as successor is in the OPEN list \
					which has a lower f than successor, skip this successor
					if(openList[idx].contains(scsrX, scsrY, successors[i].get_totalF())) continue;
					//if a node with the same position as successor is in the CLOSED list \
					which has a lower f than successor, skip this successor
					else if(closedList[idx].contains(scsrX, scsrY, successors[i].get_totalF())) continue;
					//otherwise, add the node to the open list
					else {
						Node toAdd = successors[i];
						openList[idx].addNode(toAdd);
					}
				}
			}
			//push q on the closed list
			closedList[idx].addNode(q);
			if(found == true) break;
		}
		d_Agents[idx].set_result(idx);
	}
}

int main(){
	int rows, cols;
	int noAgents;
	int choice;
	LinkedList result;

	std::cout << "Choose one" << std::endl;
	std::cout << "1: A* Search" << std::endl;
	std::cout << "2: Finite State Machine" << std::endl;
	std::cout << "3: Decision Trees" << std::endl;
	std::cin >> choice;


	if(choice == 1){
		int c = 0;
		std::cout << "1: CPU execution" << std::endl;
		std::cout << "2: CUDA execution" << std::endl;
		std::cin >> c;
		
		if(c == 1){}
		else if(c == 2) {
			const int noOfAgents = 10;
			const int noOfNodes = 64;
			size_t size = noOfAgents * sizeof(AgentSearch);
			size_t nodeSize = noOfNodes * sizeof(Node);
			std::cout << "10 Agents are being initialised" << std::endl;

			//allocate agents in host memory
			AgentSearch* h_Agents = (AgentSearch*)malloc(size);

			//allocate nodes in host memory
			Node* h_AllNodes = (Node*)malloc(nodeSize);

			//init all Nodes
			int k = 0;
			for(int i = 0; i < N; i++){
				for(int j = 0; j < N; j++){
					h_AllNodes[k].set_id(k);
					h_AllNodes[k].set_row(i);
					h_AllNodes[k].set_col(j);
					h_AllNodes[k].set_status(W);			//Walkable
					k++;
				}
			}
				
			//init agents
			//assign start and end to agents
			for(int i = 0; i < noOfAgents; i++){
				h_Agents[i].set_id(i);
				h_Agents[i].set_start(h_AllNodes[5]);	//randomise?
				h_Agents[i].set_end(h_AllNodes[25]);	//randomise?
			}

			//allocate agents in device memory
			AgentSearch* d_Agents;
			cudaMalloc((void **)&d_Agents, size);
			//copy agents from host memory to device memory
			cudaMemcpy(d_Agents, h_Agents, size, cudaMemcpyHostToDevice);

			//alocate nodes in device memory
			Node* d_AllNodes;
			cudaMalloc((void **)&d_AllNodes, nodeSize);
			//copy nodes from host memory to device memory
			cudaMemcpy(d_AllNodes, h_AllNodes, nodeSize, cudaMemcpyHostToDevice);

			//invoke kernel
			int block_size = 4;
			int n_blocks = noOfAgents/block_size + (noOfAgents%block_size == 0 ? 0:1);
			searchKernel<<<n_blocks, block_size>>>(d_Agents, d_AllNodes, noOfAgents);
			
			cudaMemcpy(h_Agents, d_Agents, sizeof(AgentSearch)*noOfAgents, cudaMemcpyDeviceToHost);

			//display results
			for(int i = 0; i < noOfAgents; i++){
				//get path for i agent
			//	result = h_Agents[i].get_path();
			//	Node n1(0,W,1,5);
			//	result.addNode(n1);
				//print path
				std::cout << h_Agents[i].get_result() << std::endl;
			//	while(result.isEmpty() == false){
					//first node from the list
			//		Node n = result.popFirstNode();
			//		std::cout << "x: " << n.get_row() << " y: " << n.get_col();
			//	}
			}

			//free memory
			free(h_Agents);
			free(h_AllNodes);
			cudaFree(d_Agents);
			cudaFree(d_AllNodes);
		}
		else
			std::cout << "ERROR, wrong number" << std::endl;
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
			int i = 0;
			const int noOfAgents = 10;
			size_t size = noOfAgents * sizeof(AgentFSM);
			std::cout << "10 Agents are being initialised" << std::endl;

			//allocate agents in host memory
			AgentFSM* h_Agents = (AgentFSM*)malloc(size);

			//populate agents
			while(i < noOfAgents){
				h_Agents[i].set_id(i);			//unique id
				h_Agents[i].set_result(0);		//0 before executions
				h_Agents[i].set_fuellvl(3);		//same for all objects for now
				h_Agents[i].set_distToCov(5);	//same for all objects for now
				h_Agents[i].set_distCovd(0);		//start 
				i++;
			}

			//allocate agents in device memory
			AgentFSM* d_Agents;
			cudaMalloc((void **)&d_Agents, size);

			//copy agents from host memory to device memory
			cudaMemcpy(d_Agents, h_Agents, size, cudaMemcpyHostToDevice);

			//invoke kernel
		//	int threadsPerBlock = 4;
		//	int blocksPerGrid = (noOfAgents + threadsPerBlock -1) / threadsPerBlock;
			int block_size = 4;
			int n_blocks = noOfAgents/block_size + (noOfAgents%block_size == 0 ? 0:1);
			//function call here
			carFSM<<<n_blocks, block_size>>>(d_Agents, noOfAgents);

			cudaMemcpy(h_Agents, d_Agents, sizeof(AgentFSM)*noOfAgents, cudaMemcpyDeviceToHost);

			for(int i = 0; i < noOfAgents; i++){
				std::cout << h_Agents[i].get_result() << std::endl;
			}

			//free memory
			free(h_Agents);
			cudaFree(d_Agents);
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
			int i = 0;
			const int noOfAgents = 10;
			size_t size = noOfAgents * sizeof(AgentDT);
			std::cout << "10 Agents are being initialised" << std::endl;

			//allocate agents in host memory
			AgentDT* h_Agents = (AgentDT*)malloc(size);
			
			//init agents
			//AgentFSM agents[noOfAgents];

			//populate agents
			while(i < noOfAgents){
				h_Agents[i].set_id(i);		//unique id
				h_Agents[i].set_result(0);	//0 before executions
				h_Agents[i].set_d1(1);		//same for all objects for now
				h_Agents[i].set_d2(2);		//same for all objects for now
				h_Agents[i].set_d3(1);		//same for all objects for now
				i++;
			}

			//allocate agents in device memory
			AgentDT* d_Agents;
			cudaMalloc((void **)&d_Agents, size);

			//copy agents from host memory to device memory
			cudaMemcpy(d_Agents, h_Agents, size, cudaMemcpyHostToDevice);

			//invoke kernel
		//	int threadsPerBlock = 4;
		//	int blocksPerGrid = (noOfAgents + threadsPerBlock -1) / threadsPerBlock;
			int block_size = 4;
			int n_blocks = noOfAgents/block_size + (noOfAgents%block_size == 0 ? 0:1);
			//function call here
			carDT<<<n_blocks, block_size>>>(d_Agents, noOfAgents);

			cudaMemcpy(h_Agents, d_Agents, sizeof(AgentDT)*noOfAgents, cudaMemcpyDeviceToHost);

			for(int i = 0; i < noOfAgents; i++){
				std::cout << h_Agents[i].get_result() << std::endl;
			}
			
			//free memory
			free(h_Agents);
			cudaFree(d_Agents);
		}
		else
			std::cout << "ERROR, wrong number" << std::endl;

	}
	else
		std::cout << "ERROR, wrong number" << std::endl;
}
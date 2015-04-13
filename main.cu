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
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>    
#include <ctime>
#include <windows.h>

#define BLOCK_SIZE 16
#define GRID_SIZE 1
const int NO_LISTS = 2;
const int N = 8;
const int W = 0; //WALKABLE
const int S = 1; //START POINT
const int E = 2; //END POINT
const int O = 3; //OBSTRUCTION

//////////////////////////////////////////////////////////// 
// Headers 
//////////////////////////////////////////////////////////// 
#ifdef _DEBUG 
#pragma comment(lib,"sfml-graphics-d.lib") 
#pragma comment(lib,"sfml-audio-d.lib") 
#pragma comment(lib,"sfml-system-d.lib") 
#pragma comment(lib,"sfml-window-d.lib") 
#pragma comment(lib,"sfml-network-d.lib") 
#else 
#pragma comment(lib,"sfml-graphics.lib") 
#pragma comment(lib,"sfml-audio.lib") 
#pragma comment(lib,"sfml-system.lib") 
#pragma comment(lib,"sfml-window.lib") 
#pragma comment(lib,"sfml-network.lib") 
#endif 
#pragma comment(lib,"opengl32.lib") 
#pragma comment(lib,"glu32.lib") 
 
#include <SFML/Graphics.hpp> 
//#include <SFML/OpenGL.hpp">
#include <iostream> 
#define _USE_MATH_DEFINES
#include <math.h>

//////////////////////////////////////////////
///////////////////SFML///////////////////////
//////////////////////////////////////////////
#define MAX_NO 4

//MAIN MENU MSGS
const char* runTestsMsg =
	"Select this option to exeucte algorithms on your device. \n"
	"Algorithms include: \n"
	"\t 1. A* Search \n"
	"\t 2. Finite State Machines \n"
	"\t 3. Decision Trees \n"
	"Following Data will be collected: \n"
	"\t 1. Execution Time \n"
	"\t 2. Speed per Watt\n"
	"\t 3. Utilization \n";

const char* viewResultsMsg = 
	"Select this option to view results of tests executed on your machine. \n";

const char* viewMyResultsMsg = 
	"Select this option to view results of tests executed on my machine. \n"
	"\nMy CPU: Intel(R) Core(TM) i5-2410 CPU @ 2.30 GHz\n"
	"My GPU: NVIDIA GeForce 410M";


//ALGORITHM MENU MSGS
const char* aStarMsg =
	"Select this option to view results for A* Search. \n";

const char* fsmMsg = 
	"Select this option to view reuslts for Finite State Machines. \n";

const char* dtMsg = 
	"Select this option to view results for Decision Trees. \n";


//RESULTS MENU MSGS
const char* executionTimeMSg =
	"Select this option to view the average execution time. \n";

const char* speedPerWattMsg = 
	"Select this option to view the average speed per watt. \n";

const char* utilizationMsg = 
	"Select this option to view the average hardware utilization. \n";


class Menu
{
public:
	Menu(float width, float height, std::string menuNames[], int options);
	~Menu(void);

	void draw(sf::RenderWindow &window);
	void MoveUp();
	void MoveDown();
	int GetPressedItem() { return selectedItemIndex; }

private:
	int selectedItemIndex;
	sf::Font font;
	sf::Text menu[MAX_NO];
};

Menu::Menu(float width, float height, std::string menuNames[], int options)
{
	if(!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf"))
	{
		//handle error
	}
	int position = 1;
	for(int i = 0; i < options; i++){
		if(i == 0){
			menu[i].setColor(sf::Color::Red);
		}
		else{
			menu[i].setColor(sf::Color::White);
		}
		menu[i].setFont(font);
		menu[i].setString(menuNames[i]);
		menu[i].setPosition(sf::Vector2f(100, height / (MAX_NO + 1) * position));
		position++;
	}

	selectedItemIndex = 0;
}


Menu::~Menu(void)
{
}


void Menu::draw(sf::RenderWindow &window){
	for(int i = 0; i < MAX_NO; i++){
		window.draw(menu[i]);
	}
}

void Menu::MoveUp(){
	if(selectedItemIndex - 1 >= 0){
		menu[selectedItemIndex].setColor(sf::Color::White);
		selectedItemIndex--;
		menu[selectedItemIndex].setColor(sf::Color::Red);

	}
}

void Menu::MoveDown(){
	if(selectedItemIndex + 1 < MAX_NO){
		menu[selectedItemIndex].setColor(sf::Color::White);
		selectedItemIndex++;
		menu[selectedItemIndex].setColor(sf::Color::Red);

	}
}

//////////////////////////////////
//////////////////////////////////
//////////////////////////////////


class Node{
public:
	int id;
	int prev;
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
	__device__ __host__ void set_prev(int cPrev){
		prev = cPrev;
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
	__device__ __host__ int get_prev() { return prev; }
	__device__ __host__ int get_status() { return status; }
	__device__ __host__ int get_row() { return row; }
	__device__ __host__ int get_col() { return col; }
	__device__ __host__ int get_costG() { return costG; }
	__device__ __host__ int get_costH() { return costH; }
	__device__ __host__ int get_totalF() { return totalF; }
	__device__ __host__ Node* get_parent() { return parent; }
};

class LinkedList{
public:
	__device__ __host__ struct Element{
			Node n1;
			Element *next;
	};

	__device__ __host__ LinkedList(){
		head = NULL;
	}

	__device__ __host__ void addNode(Node funcN){
		Element *el = new Element();
		el->n1 = funcN;

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
				lowest = cur->n1;
			}
			else if(cur->n1.get_totalF() < lowest.get_totalF()){
				lowest = cur->n1;
			}
			cur = cur->next;
			index++;
		}
		return lowest;
	}

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

	//for Node prev
	__device__ __host__ Node returnFirstNode(){
		Element *cur = head;
		Node n;
	
		if(cur != NULL){
			n = cur -> n1;
		}
	
		return n;
	}


	__device__ __host__ int removeNode(Node popNode){
		Element *cur = head, *prev;

		while(cur != NULL){
			if((cur->n1.get_row() == popNode.get_row()) && (cur->n1.get_col() == popNode.get_col())) {
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

	Element *head;
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


////////////////////////////////
//CPU///////////////////////////
////////////////////////////////
//A* Function CPU
void aStarExecutionCPU(AgentSearch* d_Agents, Node* d_AllNodes, const int noOfAgents){
	int curX, curY;
	int prevX, prevY;
	int height = 8, width = 8;
//	int G, H, F;
	//array of successors
	//init open list	
	LinkedList openList[1024];
	//init closed list
	LinkedList closedList[1024];


	bool found = false;
	
	for(int idx = 0; idx < noOfAgents; idx++){
		//end node
		Node endNode = d_Agents[idx].get_end();
		
		//should be only 4				REVIEW!!!!
		Node successors[300];

		Node n = d_Agents[idx].get_start();
		n.set_status(S);
		n.set_costG(0);
		n.set_costH(0);
		n.set_totalF(0);

		openList[idx].addNode(n);

		while(openList[idx].isEmpty() == false){
			//find the node with the least f on the open list, call it "q
			Node q = openList[idx].findLowestF();
			//pop q off the open list
			openList[idx].removeNode(q);

		/*	if(closedList[idx].contains(d_Agents[idx].get_end())){
				//path found
				found = true;
				break;
			}
*/
			curX = q.get_row();
			curY = q.get_col();

			int k=0;
			//generate q's 4successors and set their parents to q
			for(int i = curX - 1; i <= curX + 1; ++i){
				for(int j = curY - 1; j <= curY + 1; ++j){
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
			for(int j = 0; j <= 3; j++){
				//get successors X & Y
				int scsrX = successors[j].get_row();
				int scsrY = successors[j].get_col();
				int status = successors[j].get_status();
				//if successor is the goal, stop the search
			//	if((scsrX == d_Agents[idx].get_end().get_row()) && (scsrY == d_Agents[idx].get_end().get_col())){
				if((scsrX == endNode.get_row()) && (scsrY == endNode.get_col())){
					//stop search
					found = true;
					break;
				}
				//out of grid
				else if((scsrX > 7) || (scsrY > 7) || ((scsrX < 0) ||(scsrY < 0))) continue;
				else if(d_AllNodes[idx * height * width + scsrX * width + scsrY].get_status() == O) continue;
				else{
					//successor.g = q.g + distance between successor and q
					//successors[i].set_costG(q.get_costG() +(std::abs(curX - d_Agents[idx].get_start().get_row()) + std::abs(curY - d_Agents[idx].get_start().get_col())));
					int G = std::abs(curX - n.get_row()) + std::abs(curY - n.get_col());
					successors[j].set_costG(G);
					//successors[i].set_costG(q.get_costG() + 10);
					
					//successor.h = distance from goal to successor
				//	int H = std::abs(curX - endNode.get_row()) + std::abs(curY - endNode.get_col());
					int H = std::abs(scsrX - endNode.get_row()) + std::abs(scsrY - endNode.get_col());
					successors[j].set_costH(H);
					
					//successor.f = successor.g + successor.h
					int F = successors[j].get_costG() + successors[j].get_costH();
					successors[j].set_totalF(F);

					//if a node with the same position as successor is in the OPEN list \
					which has a lower f than successor, skip this successor
					if(openList[idx].contains(scsrX, scsrY, successors[j].get_totalF())) continue;
					//if a node with the same position as successor is in the CLOSED list \
					which has a lower f than successor, skip this successor
					else if(closedList[idx].contains(scsrX, scsrY, successors[j].get_totalF())) continue;
					//otherwise, add the node to the open list
					else {
						Node toAdd = successors[j];
						openList[idx].addNode(toAdd);
					}
				}
			}
			//push q on the closed list
			closedList[idx].addNode(q);

			
			//if empty set prev to zero
			if(closedList[idx].isEmpty() == true){
			//	prevX = d_Agents[idx].get_start().row;
			//	prevY = d_Agents[idx].get_start().col;
				prevX = 0;
				prevY = 0;
				
			}
			else{
				Node prev = closedList[idx].returnFirstNode();
				prevX = prev.get_row();
				prevY = prev.get_col();
			}
			
			int curId = idx * height * width + q.get_row()* width + q.get_col();
			int prevId;
			if(prevX == 0 && prevY == 0){
				prevId = 0;
			}
			else{
				prevId = idx * height * width + prevX * width + prevY;
			}
			d_AllNodes[curId].set_prev(prevId);
			if(found == true) break;
		}
	}
}

//single execution of a star search
void singleExeAStar(){
	int height = 8, width = 8;
	int noOfAgents = 1;
	//grid of 8x8 = 64 nodes
	const int noOfNodes = height * width;
	size_t size = noOfAgents * sizeof(AgentSearch);
	//get the size of all nodes of all agents
	size_t allNodesSize = noOfAgents * (noOfNodes * sizeof(Node));

	int startPath = 61;
	int endPath = 40; 

	//std::cout << noOfAgents << " Agents are being initialised" << std::endl;

	//allocate agents in host memory
	AgentSearch* h_Agents = (AgentSearch*)malloc(size);

	//allocate all nodes in host memory
	Node* h_AllNodes = (Node*)malloc(allNodesSize);

	//init all Nodes flat 3D array
	int l = 0;
	for(int k = 0; k < noOfAgents; k++){
		for(int r = 0; r < height; r++){
			for(int c = 0; c < width; c++){
				h_AllNodes[k*height*width + r*width + c].set_id(l);
				h_AllNodes[k*height*width + r*width + c].set_row(r);
				h_AllNodes[k*height*width + r*width + c].set_col(c);
				h_AllNodes[k*height*width + r*width + c].set_status(W); //walkable
			}
		}
	}

//	h_AllNodes[21].set_status(O);

	//init agents
	//assign start and end to agents
	for(int i = 0; i < noOfAgents; i++){
		h_Agents[i].set_id(i);
		h_Agents[i].set_start(h_AllNodes[startPath]);	//randomise?
		h_Agents[i].set_end(h_AllNodes[endPath]);	//randomise?
	}

	//call function
	aStarExecutionCPU(h_Agents, h_AllNodes, noOfAgents);

	//print results
	for(int i =0; i < 64; i++){
		std::cout << "Index " << i << ": " << h_AllNodes[i].get_row() << " " << h_AllNodes[i].get_col() << " Prev id: " << h_AllNodes[i].get_prev() << std::endl;
	}
			

	//free memory
	free(h_Agents);
	free(h_AllNodes);
}

//A* Setup CPU
void aStarSearchSetupCPU(int watts){
	int height = 8, width = 8;
	const int noOfAgentsArr[6] = {32, 64, 128, 256, 512, 1024};
	for(int i = 0; i < 6; i++){
		const int noOfAgents = noOfAgentsArr[i];
		for(int j = 0; j < 100; j++){
			//for execution time
			//http://stackoverflow.com/questions/14337278/precise-time-measurement
			LARGE_INTEGER frequency;        // ticks per second
			LARGE_INTEGER t1, t2;           // ticks
			double elapsedTime;

			//grid of 8x8 = 64 nodes
			const int noOfNodes = height * width;
			size_t size = noOfAgents * sizeof(AgentSearch);
			//get the size of all nodes of all agents
			size_t allNodesSize = noOfAgents * (noOfNodes * sizeof(Node));

			int startPath = 20;
			int endPath = 47;
	
	//		std::cout << noOfAgents << " Agents are being initialised" << std::endl;

			//allocate agents in host memory
			AgentSearch* h_Agents = (AgentSearch*)malloc(size);

			//allocate all nodes in host memory
			Node* h_AllNodes = (Node*)malloc(allNodesSize);

			//init all Nodes flat 3D array
			int l = 0;
			for(int k = 0; k < noOfAgents; k++){
				for(int r = 0; r < height; r++){
					for(int c = 0; c < width; c++){
						h_AllNodes[k*height*width + r*width + c].set_id(l);
						h_AllNodes[k*height*width + r*width + c].set_row(r);
						h_AllNodes[k*height*width + r*width + c].set_col(c);
						h_AllNodes[k*height*width + r*width + c].set_status(W); //walkable
					}
				}
			}

		//	h_AllNodes[21].set_status(O);

			//init agents
			//assign start and end to agents
			for(int i = 0; i < noOfAgents; i++){
				h_Agents[i].set_id(i);
				h_Agents[i].set_start(h_AllNodes[startPath]);	//randomise?
				h_Agents[i].set_end(h_AllNodes[endPath]);	//randomise?
			}

			// get ticks per second
			QueryPerformanceFrequency(&frequency);
			// start timer
			QueryPerformanceCounter(&t1);

			//call function
			aStarExecutionCPU(h_Agents, h_AllNodes, noOfAgents);

			//stop timer
			QueryPerformanceCounter(&t2);

			// compute and print the elapsed time in millisec
			elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;

			//print results
		//	for(int i =0; i < 64*noOfAgents; i++){
		//		std::cout << "Index " << i << ": " << h_AllNodes[i].get_row() << " " << h_AllNodes[i].get_col() << " Prev id: " << h_AllNodes[i].get_prev() << std::endl;
		//	}
			
			//print execution time
		//	std::cout << "Time for A*: " << elapsedTime << std::endl;

			//save execution time to a CSV file
			std::ofstream myfile;
			myfile.open("CPUastarExecutionTime.csv",std::ios_base::app);
			myfile << noOfAgents << ", " << elapsedTime << "\n";
			myfile.close();

			std::ofstream myfileWatts;
			myfileWatts.open("CPUastarSpeedPerWatts.csv",std::ios_base::app);
			myfileWatts << noOfAgents << ", " << (double)watts/noOfAgents << "\n";
			myfileWatts.close();

			//free memory
			free(h_Agents);
			free(h_AllNodes);
		}
	}
}

//FSM Function CPU
void fsmExecutionCPU(AgentFSM* agents,  int noOfAgents){
	int myDistCovd;
	int myFuelLvl;
	const int startState = 1;
	const int driveState = 2;
	const int getFuelState = 3;
	const int endState = 4;
	
	for(int i = 0; i < noOfAgents; i++){
		bool finished = false;
		int index = 0;
		agents[i].set_currentState(startState); //store current state
		index++;

		while(index < 30){
			switch(agents[i].get_currentState()){
				//START STATE
				case startState:
				//	std::cout << "Start State" << std::endl;
					agents[i].set_currentState(driveState);
					index++;
					break;
				//DRIVING STATE
				case driveState:
					//check if the distance covered so far is not the goal distance
					if(agents[i].get_distCovd() != agents[i].get_distToCov()){
						//check if the car needs to get fuel
						if(agents[i].get_fuellvl() < 2){
							agents[i].set_currentState(getFuelState);
							index++;
						}
						//proceed if theres more distance to cover & car has fuel
						else{
						//	std::cout << "Driving State" << std::endl;
							agents[i].set_currentState(driveState);
							//update distance covered
							myDistCovd = agents[i].get_currentState();
							myDistCovd++;	//distance + 1
							agents[i].set_distCovd(myDistCovd);
							//update fuel level
							myFuelLvl = agents[i].get_fuellvl();
							myFuelLvl--;	//fuel - 1
							agents[i].set_fuellvl(myFuelLvl);
							index++;
						}
					}
					//when at goal
					else{
						agents[i].set_currentState(endState);
						index++;
					}
					break;
				//GETTING FUEL
				case getFuelState:
				//	std::cout << "Get Fuel State" << std::endl;
					//fuel up by 1
					myFuelLvl = agents[i].get_fuellvl();
					myFuelLvl += 1;
					agents[i].set_fuellvl(myFuelLvl);
					agents[i].set_currentState(driveState);
					index++;
					break;
				//END STATE --> at destination
				case endState:
				//	std::cout << "End State" << std::endl;
					agents[i].set_currentState(endState);
					//end while loop 
					finished = true;
					index++;
					break;
			}
		}
		//number of states
		agents[i].set_result(index);
	}
}

//FSM Setup CPU
void fsmSetupCPU(int watts){
	int i = 0;
	const int noOfAgentsArr[6] = {32, 64, 128, 256, 512, 1024};
	for(int i = 0; i < 6; i++){
		const int noOfAgents = noOfAgentsArr[i];
		for(int j = 0; j < 100; j++){
			//for execution time
			//http://stackoverflow.com/questions/14337278/precise-time-measurement
			LARGE_INTEGER frequency;        // ticks per second
			LARGE_INTEGER t1, t2;           // ticks
			double elapsedTime;

			size_t size = noOfAgents * sizeof(AgentFSM);
	//		std::cout << noOfAgents << " Agents are being initialised" << std::endl;

			//allocate agents in host memory
			AgentFSM* agents = (AgentFSM*)malloc(size);

			//populate agents
			for(int i = 0; i < noOfAgents; i++){
				agents[i].set_id(i);			//unique id
				agents[i].set_result(0);		//0 before executions
				agents[i].set_fuellvl(3);		//same for all objects for now
				agents[i].set_distToCov(5);	//same for all objects for now
				agents[i].set_distCovd(0);	//start 
			}
	
			// get ticks per second
			QueryPerformanceFrequency(&frequency);
			// start timer
			QueryPerformanceCounter(&t1);

			//call function
			fsmExecutionCPU(agents, noOfAgents);

			//stop timer
			QueryPerformanceCounter(&t2);

			// compute and print the elapsed time in millisec
			elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;

		//	for(int i = 0; i < noOfAgents; i++){
		//		std::cout << h_Agents[i].get_result() << std::endl;
		//	}

		//	std::cout << "FSM Time: " << elapsedTime << std::endl;

			//save execution time to a CSV file
			std::ofstream myfile;
			myfile.open("CPUfsmExecutionTime.csv",std::ios_base::app);
			myfile << noOfAgents << ", " << elapsedTime << "\n";
			myfile.close();

			std::ofstream myfileWatts;
			myfileWatts.open("CPUfsmWatts.csv",std::ios_base::app);
			myfileWatts << noOfAgents << ", " << (double)watts/noOfAgents << "\n";
			myfileWatts.close();
			
			//free memory
			free(agents);
		}
	}
}


//DT Function CPU
void dtExecutionCPU(AgentDT* d_array,  int noOfAgents){
//	int idx = blockIdx.x*blockDim.x + threadIdx.x;
/*	srand( time(NULL) ); 
	int atDest = rand()%2;
	int parkAv = rand()%2;
	int lowPetrol = rand()%2;*/
	/* 1 - You are at destination. Car Parked
	*  2 - You are at destination. Looking for parking
	*  3 - Not at destination. Low petrol - looking for a petrol station
	*  4 - Not at destination. Still driving
	*/

	for(int i = 0; i < noOfAgents; i++){
		//if at destination
		if(d_array[i].get_d1() == 1){
			//if parking available
			if(d_array[i].get_d2() == 1){
		//		std::cout << "You are at destination. Car Parked." << std::endl;
				d_array[i].set_result(1);
			}
			//if parking unavailable
			else{
		//		std::cout << "You are at destination. Looking for parking." << std::endl;
				d_array[i].set_result(2);
			}
		}
		//if not at destination
		else{
			//if low on petrol
			if(d_array[i].get_d2() == 1){
			//	std::cout << "Not at destination. Low petrol - looking for a petrol station." << std::endl;
				d_array[i].set_result(3);
			}
			//if not low on petrol
			else{
			//	std::cout << "Not at destination. Still driving." << std::endl;
				d_array[i].set_result(4);
			}
		}
	}
}

//DT Setup CPU
void dtSetupCPU(int watts){
	int i = 0;
	const int noOfAgentsArr[6] = {32, 64, 128, 256, 512, 1024};
	for(int i = 0; i < 6; i++){
		for(int j = 0; j < 100; j++){
			//for execution time
			//http://stackoverflow.com/questions/14337278/precise-time-measurement
			LARGE_INTEGER frequency;        // ticks per second
			LARGE_INTEGER t1, t2;           // ticks
			double elapsedTime;

			//for allocation
			const int noOfAgents = noOfAgentsArr[i];
			size_t size = noOfAgents * sizeof(AgentDT);
//			std::cout << noOfAgents << " Agents are being initialised" << std::endl;

			//allocate agents in host memory
			AgentDT* agents = (AgentDT*)malloc(size);
			
			//populate agents
			for(int i = 0; i < noOfAgents; i++){
				agents[i].set_id(i);		//unique id
				agents[i].set_result(0);	//0 before executions
				agents[i].set_d1(1);		//same for all objects for now
				agents[i].set_d2(2);		//same for all objects for now
				agents[i].set_d3(1);		//same for all objects for now
			}

			// get ticks per second
			QueryPerformanceFrequency(&frequency);
			// start timer
			QueryPerformanceCounter(&t1);

			//call function
			dtExecutionCPU(agents, noOfAgents);

			//stop timer
			QueryPerformanceCounter(&t2);

			// compute and print the elapsed time in millisec
			elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;

	
		//	for(int i = 0; i < noOfAgents; i++){
		//		std::cout << h_Agents[i].get_result() << std::endl;
		//	}
	
		//	std::cout << "DT Time: " << cpuElapseTime << std::endl;


			//save execution time to a CSV file
			std::ofstream myfile;
			myfile.open("CPUdtExecutionTime.csv",std::ios_base::app);
			myfile << noOfAgents << ", " << elapsedTime << "\n";
			myfile.close();

			double wattsResult = (double)watts/noOfAgents;

			std::ofstream myfileWatts;
			myfileWatts.open("CPUdtWatts.csv",std::ios_base::app);
			myfileWatts << noOfAgents << ", " << wattsResult << "\n";
			myfileWatts.close();

			//free memory
			free(agents);
		}
	}
}


////////////////////////////////
////////////CUDA////////////////
////////////////////////////////
//A* Function CUDA
__global__ void aStarExecutionCUDA(AgentSearch* d_Agents, Node* d_AllNodes, const int noOfAgents){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int curX, curY;
	int prevX, prevY;
	int height = 8, width = 8;
	int G, H, F;
	//array of successors
	//init open list	
	LinkedList openList[1024];
	//init closed list
	LinkedList closedList[1024];


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
			//generate q's 4successors and set their parents to q
			for(int i = curX - 1; i <= curX + 1; ++i){
				for(int j = curY - 1; j <= curY + 1; ++j){
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
				int status = successors[i].get_status();
				//if successor is the goal, stop the search
				if((scsrX == d_Agents[idx].get_end().get_row()) && (scsrY == d_Agents[idx].get_end().get_col())){
					//stop search
					found = true;
					break;
				}
				//out of grid
				else if((scsrX > 7) || (scsrY > 7)) continue;
				else if(d_AllNodes[idx * height * width + scsrX * width + scsrY].get_status() == O) continue;
				else{
					//successor.g = q.g + distance between successor and q
					//successors[i].set_costG(q.get_costG() +(std::abs(curX - d_Agents[idx].get_start().get_row()) + std::abs(curY - d_Agents[idx].get_start().get_col())));
					successors[i].set_costG(std::abs(curX - d_Agents[idx].get_start().get_row()) + std::abs(curY - d_Agents[idx].get_start().get_col()));
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
			
			//if empty set prev to zero
			if(closedList[idx].isEmpty() == true){
			//	prevX = d_Agents[idx].get_start().row;
			//	prevY = d_Agents[idx].get_start().col;
				prevX = 0;
				prevY = 0;
				
			}
			else{
				Node prev = closedList[idx].returnFirstNode();
				prevX = prev.get_row();
				prevY = prev.get_col();
			}
			//push q on the closed list
			closedList[idx].addNode(q);
			int curId = idx * height * width + q.get_row()* width + q.get_col();
			int prevId;
			if(prevX == 0 && prevY == 0){
				prevId = 0;
			}
			else{
				prevId = idx * height * width + prevX * width + prevY;
			}
			d_AllNodes[curId].set_prev(prevId);
			if(found == true) break;
		}
	}
}

//A* Setup CUDA
void aStarSearchSetupCUDA(int watts){
	int height = 8, width = 8;
	const int noOfAgentsArr[6] = {32, 64, 128, 256, 512, 1024};
	for(int i = 0; i < 6; i++){
		for(int j = 0; j < 100; j++){
			//used to implement performance metrics
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			//number of threads (idx on cuda)
			const int noOfAgents = noOfAgentsArr[i];
			//grid of 8x8 = 64 nodes
			const int noOfNodes = height * width;
			size_t size = noOfAgents * sizeof(AgentSearch);
			//get the size of all nodes of all agents
			size_t allNodesSize = noOfAgents * (noOfNodes * sizeof(Node));

			int startPath = 13;
			int endPath = 63;
	
		//	std::cout << noOfAgents << " Agents are being initialised" << std::endl;

			//allocate agents in host memory
			AgentSearch* h_Agents = (AgentSearch*)malloc(size);
			//allocate all nodes in host memory
			Node* h_AllNodes = (Node*)malloc(allNodesSize);

			//init all Nodes flat 3D array
			int l = 0;
			for(int k = 0; k < noOfAgents; k++){
				for(int r = 0; r < height; r++){
					for(int c = 0; c < width; c++){
						h_AllNodes[k*height*width + r*width + c].set_id(l);
						h_AllNodes[k*height*width + r*width + c].set_row(r);
						h_AllNodes[k*height*width + r*width + c].set_col(c);
						h_AllNodes[k*height*width + r*width + c].set_status(W); //walkable
					}
				}
			}

		//	h_AllNodes[21].set_status(O);

			//init agents
			//assign start and end to agents
			for(int i = 0; i < noOfAgents; i++){
				h_Agents[i].set_id(i);
				h_Agents[i].set_start(h_AllNodes[startPath]);	//randomise?
				h_Agents[i].set_end(h_AllNodes[endPath]);	//randomise?
			}

			//allocate agents in device memory
			AgentSearch* d_Agents;
			cudaMalloc((void **)&d_Agents, size);
			//copy agents from host memory to device memory
			cudaMemcpy(d_Agents, h_Agents, size, cudaMemcpyHostToDevice);

			//alocate nodes in device memory
			Node* d_AllNodes;
			cudaMalloc((void **)&d_AllNodes, allNodesSize);
			//copy nodes from host memory to device memory
			cudaMemcpy(d_AllNodes, h_AllNodes, allNodesSize, cudaMemcpyHostToDevice);

			//record execution time
			cudaEventRecord(start,0);	
			//invoke kernel
			aStarExecutionCUDA<<<1, noOfAgents>>>(d_Agents, d_AllNodes, noOfAgents);
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);

			//copy results back to the host array of nodes
			cudaMemcpy(h_AllNodes, d_AllNodes, allNodesSize, cudaMemcpyDeviceToHost);

			//store execution time in variable milliseconds
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);

			//destroy events
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			//reset the gpu
			cudaDeviceReset();

			/*
			//print results
			for(int i =0; i < 64*noOfAgents; i++){
				std::cout << "Index " << i << ": " << h_AllNodes[i].get_row() << " " << h_AllNodes[i].get_col() << " Prev id: " << h_AllNodes[i].get_prev() << std::endl;
			}
			*/
			//print execution time
		//	std::cout << "Time for the kernel: %f ms\n: " << milliseconds << std::endl;

			//save execution time to a CSV file
			std::ofstream myfile;
			myfile.open("CUDAastarExecutionTime.csv",std::ios_base::app);
			myfile << noOfAgents << ", " << milliseconds << "\n";
			myfile.close();

			std::ofstream myfileWatts;
			myfileWatts.open("CUDAastarWatts.csv",std::ios_base::app);
			myfileWatts << noOfAgents << ", " << (double)watts/noOfAgents << "\n";
			myfileWatts.close();

			//free memory
			free(h_Agents);
			free(h_AllNodes);
			cudaFree(d_Agents);
			cudaFree(d_AllNodes);
		}
	}
}


//FSM Function CUDA
//int tankSize, int distToCover <-- randomise? 
__global__ void fsmExecutionCUDA(AgentFSM* d_array,  int noOfAgents){
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

//FSM Setup CUDA
void fsmSetupCUDA(int watts){
	int i = 0;
	const int noOfAgentsArr[6] = {32, 64, 128, 256, 512, 1024};
	for(int i = 0; i < 6; i++){
		const int noOfAgents = noOfAgentsArr[i];
		for(int j = 0; j < 100; j++){
			std::cout << noOfAgents << std::endl;
			//used to implement performance metrics
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			size_t size = noOfAgents * sizeof(AgentFSM);
		//	std::cout << noOfAgents << " Agents are being initialised" << std::endl;

			//allocate agents in host memory
			AgentFSM* h_Agents = (AgentFSM*)malloc(size);

			//populate agents
			for(int i = 0; i < noOfAgents; i++){
				h_Agents[i].set_id(i);			//unique id
				h_Agents[i].set_result(0);		//0 before executions
				h_Agents[i].set_fuellvl(3);		//same for all objects for now
				h_Agents[i].set_distToCov(5);	//same for all objects for now
				h_Agents[i].set_distCovd(0);	 
			}

			//allocate agents in device memory
			AgentFSM* d_Agents;
			cudaMalloc((void **)&d_Agents, size);

			//copy agents from host memory to device memory
			cudaMemcpy(d_Agents, h_Agents, size, cudaMemcpyHostToDevice);

	
			cudaEventRecord(start,0);	
			//invoke kernel
			fsmExecutionCUDA<<<1, noOfAgents>>>(d_Agents, noOfAgents);
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);

			//copy the results back
			cudaMemcpy(h_Agents, d_Agents, sizeof(AgentFSM)*noOfAgents, cudaMemcpyDeviceToHost);

			//store execution time in variable milliseconds
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);

			//destroy events
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			//reset the gpu
			cudaDeviceReset();

		//	for(int i = 0; i < noOfAgents; i++){
		//		std::cout << h_Agents[i].get_result() << std::endl;
		//	}

		//	std::cout << "FSM Time: " << milliseconds << std::endl;

			//save execution time to a CSV file
			std::ofstream myfile;
			myfile.open("CUDAfsmExecutionTime.csv",std::ios_base::app);
			myfile << noOfAgents << ", " << milliseconds << "\n";
			myfile.close();

			std::ofstream myfileWatts;
			myfileWatts.open("CUDAfsmWatts.csv",std::ios_base::app);
			myfileWatts << noOfAgents << ", " << (double)watts/noOfAgents << "\n";
			myfileWatts.close();

			//free memory
			free(h_Agents);
			cudaFree(d_Agents);
		}
	}
}


//DT Function CUDA
__global__ void dtExecutionCUDA(AgentDT* d_array,  int noOfAgents){
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

//DT Setup CUDA
void dtSetupCUDA(int watts){
	int i = 0;
	const int noOfAgentsArr[6] = {32, 64, 128, 256, 512, 1024};
	for(int i = 0; i < 6; i++){
		const int noOfAgents = noOfAgentsArr[i];
		for(int j = 0; j < 100; j++){
			//used to implement performance metrics
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			//for allocation
			size_t size = noOfAgents * sizeof(AgentDT);
		//	std::cout << noOfAgents << " Agents are being initialised" << std::endl;

			//allocate agents in host memory
			AgentDT* h_Agents = (AgentDT*)malloc(size);
			
			//init agents
			//AgentFSM agents[noOfAgents];

			//populate agents
			for(int i = 0; i < noOfAgents; i++){
				h_Agents[i].set_id(i);		//unique id
				h_Agents[i].set_result(0);	//0 before executions
				h_Agents[i].set_d1(1);		//same for all objects for now
				h_Agents[i].set_d2(2);		//same for all objects for now
				h_Agents[i].set_d3(1);		//same for all objects for now
			}

			//allocate agents in device memory
			AgentDT* d_Agents;
			cudaMalloc((void **)&d_Agents, size);

			//copy agents from host memory to device memory
			cudaMemcpy(d_Agents, h_Agents, size, cudaMemcpyHostToDevice);

			cudaEventRecord(start,0);	
			//invoke kernel
			dtExecutionCUDA<<<1, noOfAgents>>>(d_Agents, noOfAgents);
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);

			cudaMemcpy(h_Agents, d_Agents, sizeof(AgentDT)*noOfAgents, cudaMemcpyDeviceToHost);

			//store execution time in variable milliseconds
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);

			//destroy events
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			//reset the gpu
			cudaDeviceReset();
	
	
		//	for(int i = 0; i < noOfAgents; i++){
		//		std::cout << h_Agents[i].get_result() << std::endl;
		//	}
	
		//	std::cout << "DT Time: " << milliseconds << std::endl;


			//save execution time to a CSV file
			std::ofstream myfile;
			myfile.open("dtExecutionTime.csv",std::ios_base::app);
			myfile << noOfAgents << ", " << milliseconds << "\n";
			myfile.close();

			std::ofstream myfileWatts;
			myfileWatts.open("CUDAdtWatts.csv",std::ios_base::app);
			myfileWatts << noOfAgents << ", " << (double)watts/noOfAgents << "\n";
			myfileWatts.close();


			//free memory
			free(h_Agents);
			cudaFree(d_Agents);
		}
	}
}


//READ CSV FUNCTION RETURNS THE EXECUTION AVG TIME 
//option = 2 -> my results
//option = 3 -> user results
//algo 1 = A*, 2 = FSM, 3 = DT
float* readCSV(std::string fileName){
	
	//read from csv
	std::ifstream myfile(fileName);
//	myfile.open(fileName);

	//function returns results array 
	float* results = 0;
	results = new float[6];

	//if file name is valid (exists) then proceed
	if(myfile){
		//2d array for data [number of agents] [data]
		float** data = 0;
		data = new float*[1000];

		for(int row = 0; row < 1000; ++row){
			data[row] = new float[2];
			std::string line;
			std::getline(myfile, line);

			if(!myfile.good()) break;

			std::stringstream ss(line);
						
			for(int col =0; col < 2; ++col){
				std::string val;
				std::getline(ss, val, ',');
							
				if(!ss) break;

				std::stringstream convertor(val);
				convertor >> data[row][col];
			}
		}


		float avg = 0;
		float total = 0;
		//loop through agents
		for(int i = 0; i < 6; i++){

			int j = 0 + 100*(i+1);
			for(int k = j-100; k < j; k++){
				//got total of data for a certain number of agents
				total = total + data[k][1];
			}
			//get avarage
			avg = total / 100;
			results[i] = avg;

			avg = 0;
			total = 0;
		}


		return results;
	}
	//if file does not exist
	else{
		results[0] = -1;
		return results;
	}
}

void runTests(int cpuWatts, int gpuWatts){	
/*	int nDevices;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaGetDeviceProperties(&prop, i);
	/*	printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
				prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
				prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		printf("threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
	} 
	*/

	aStarSearchSetupCUDA(gpuWatts);
	fsmSetupCUDA(gpuWatts);
	dtSetupCUDA(gpuWatts); 
	aStarSearchSetupCPU(cpuWatts);
	fsmSetupCPU(cpuWatts);
	dtSetupCPU(cpuWatts);
	std::cout << cpuWatts << " " << gpuWatts << std::endl;
}


//return circle objects to present data
//option - 1 = cpu, 2 = cuda (different colour)
sf::CircleShape* drawData(sf::VertexArray xAxisPos, float* data, int option){
	int x, y;
	sf::CircleShape* points;
	points = new sf::CircleShape[6];

	for(int i = 0; i < 6; i++){
		x = (xAxisPos[0].position.x + 100 * (i+1)) - 2; 
		y = xAxisPos[0].position.y - data[i] - 10; 

		points[i].setPosition(x, y);
		points[i].setRadius(2);
		if(option == 1)
			points[i].setFillColor(sf::Color::Red);
		else if(option == 2)
			points[i].setFillColor(sf::Color::Green);
	}

	return points;
}

int main(){
	int height = 8, width = 8;
	int noAgents;
	int choice;

    // Create the main window 
    sf::RenderWindow window(sf::VideoMode(1200, 600, 32), "CUDA Benchamrk"); 
	
	//menu headings
	std::string mainMenu[4] = {"Test on your device", "View your results", "View my results", "Exit"};
	std::string algorithmMenu[4] = {"A* Search", "Finite State Machine", "Decision Tree", "Return to Main Menu"};
	std::string resultMenu[4] =  {"Execution time", "Speed per Watt", "Utilization", "Return to Algorithm Menu"};
	
	//init menus
	Menu menu(window.getSize().x, window.getSize().y, mainMenu, 4);
	Menu algoMenu(window.getSize().x, window.getSize().y, algorithmMenu, 4);
	Menu resMenu(window.getSize().x, window.getSize().y, resultMenu, 4);

	//init font
	sf::Font font;
	font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf");

	//text msg
	sf::Text displayText("",font);
	displayText.setCharacterSize(20);
	displayText.setPosition(500, 120);
	int mainMenuCounter = 1;
	int algoMenuCounter = 1;
	int resMenuCounter = 1;

	//x axis
	sf::VertexArray xAxis(sf::LinesStrip, 2);
	xAxis[0].position = sf::Vector2f(100, 500);
	xAxis[1].position = sf::Vector2f(720, 500);

	//y axis
	sf::VertexArray yAxis(sf::LinesStrip, 2);
	yAxis[0].position = sf::Vector2f(100, 100);
	yAxis[1].position = sf::Vector2f(100, 500);

	
	//y axis lines
	sf::VertexArray line1(sf::LinesStrip, 2);
	line1[0].position = sf::Vector2f(95, 450);
	line1[1].position = sf::Vector2f(105, 450);
	
	sf::VertexArray line2(sf::LinesStrip, 2);
	line2[0].position = sf::Vector2f(95, 400);
	line2[1].position = sf::Vector2f(105, 400);

	sf::VertexArray line3(sf::LinesStrip, 2);
	line3[0].position = sf::Vector2f(95, 350);
	line3[1].position = sf::Vector2f(105, 350);

	sf::VertexArray line4(sf::LinesStrip, 2);
	line4[0].position = sf::Vector2f(95, 300);
	line4[1].position = sf::Vector2f(105, 300);

	sf::VertexArray line5(sf::LinesStrip, 2);
	line5[0].position = sf::Vector2f(95, 250);
	line5[1].position = sf::Vector2f(105, 250);

	sf::VertexArray line6(sf::LinesStrip, 2);
	line6[0].position = sf::Vector2f(95, 200);
	line6[1].position = sf::Vector2f(105, 200);

	sf::VertexArray line7(sf::LinesStrip, 2);
	line7[0].position = sf::Vector2f(95, 150);
	line7[1].position = sf::Vector2f(105, 150);

	sf::VertexArray line8(sf::LinesStrip, 2);
	line8[0].position = sf::Vector2f(95, 100);
	line8[1].position = sf::Vector2f(105, 100);


	//x axis
	sf::VertexArray xline1(sf::LinesStrip, 2);
	xline1[0].position = sf::Vector2f(200, 495);
	xline1[1].position = sf::Vector2f(200, 505);
	
	sf::VertexArray xline2(sf::LinesStrip, 2);
	xline2[0].position = sf::Vector2f(300, 495);
	xline2[1].position = sf::Vector2f(300, 505);

	sf::VertexArray xline3(sf::LinesStrip, 2);
	xline3[0].position = sf::Vector2f(400, 495);
	xline3[1].position = sf::Vector2f(400, 505);

	sf::VertexArray xline4(sf::LinesStrip, 2);
	xline4[0].position = sf::Vector2f(500, 495);
	xline4[1].position = sf::Vector2f(500, 505);

	sf::VertexArray xline5(sf::LinesStrip, 2);
	xline5[0].position = sf::Vector2f(600, 495);
	xline5[1].position = sf::Vector2f(600, 505);
	
	sf::VertexArray xline6(sf::LinesStrip, 2);
	xline6[0].position = sf::Vector2f(700, 495);
	xline6[1].position = sf::Vector2f(700, 505);

	//circle shapes for graphs
	sf::CircleShape* cpuPoints;
	sf::CircleShape* cudaPoints;

	//title
	sf::Text title;
	title.setFont(font);
	title.setStyle(sf::Text::Underlined| sf::Text::Italic | sf::Text::Bold);
	title.setPosition(300,50);
	title.setCharacterSize(30); 

	//time text
	sf::Text time;
	time.setFont(font);
	time.setString("Time");
	time.setStyle(sf::Text::Bold);
	time.setPosition(30, 300);
	time.setCharacterSize(16);

	//no of agents text
	sf::Text agents;
	agents.setFont(font);
	agents.setString("Number of agents");
	agents.setStyle(sf::Text::Bold);
	agents.setPosition(350, 550);
	agents.setCharacterSize(16);


	//run tests text
	sf::Text cpuWattsText("CPU Watts",font);
	cpuWattsText.setCharacterSize(20);
	cpuWattsText.setPosition(20, 200);

	sf::Text gpuWattsText("GPU Watts",font);
	gpuWattsText.setCharacterSize(20);
	gpuWattsText.setPosition(20, 250);
	

	//watts counters
	int cpuWtsC = 10;
	int gpuWtsC = 10;

	sf::Text cpuWtsCText("",font);
	cpuWtsCText.setCharacterSize(20);
	cpuWtsCText.setPosition(210, 200);

	sf::Text gpuWtsCText("",font);
	gpuWtsCText.setCharacterSize(20);
	gpuWtsCText.setPosition(210, 250);

	//+ & - buttons for number of watts
	//CPU MINUS
	sf::RectangleShape cpuMinus;
	cpuMinus.setSize(sf::Vector2f(40, 20));
	cpuMinus.setOutlineColor(sf::Color::Red);
	cpuMinus.setOutlineThickness(1);
	cpuMinus.setPosition(150, 200);
	
	//GPU MINUS
	sf::RectangleShape gpuMinus;
	gpuMinus.setSize(sf::Vector2f(40, 20));
	gpuMinus.setOutlineColor(sf::Color::Red);
	gpuMinus.setOutlineThickness(1);
	gpuMinus.setPosition(150, 250);

	//CPU PLUS
	sf::RectangleShape cpuPlus;
	cpuPlus.setSize(sf::Vector2f(40, 20));
	cpuPlus.setOutlineColor(sf::Color::Red);
	cpuPlus.setOutlineThickness(1);
	cpuPlus.setPosition(250, 200);

	//GPU PLUS
	sf::RectangleShape gpuPlus;
	gpuPlus.setSize(sf::Vector2f(40, 20));
	gpuPlus.setOutlineColor(sf::Color::Red);
	gpuPlus.setOutlineThickness(1);
	gpuPlus.setPosition(250, 250);

	int option = 0;
	int algoOption = 0;
	int resOption = 0;
	bool testsRunning = false;
	int x, y;		
	int x1, y1;

    // Start game loop 
    while (window.isOpen()) 
    { 
        // Process events 
        sf::Event event; 
        while (window.pollEvent(event)) 
        {
			if(sf::Mouse::isButtonPressed(sf::Mouse::Left))
			{
				// transform the mouse position from window coordinates to world coordinates
				sf::Vector2f mouse = window.mapPixelToCoords(sf::Mouse::getPosition(window));

				// retrieve the bounding box of the sprite
				sf::FloatRect cpuMinBounds = cpuMinus.getGlobalBounds();
				sf::FloatRect cpuPlusBounds = cpuPlus.getGlobalBounds();
				sf::FloatRect gpuMinBounds = gpuMinus.getGlobalBounds();
				sf::FloatRect gpuPlusBounds = gpuPlus.getGlobalBounds();

				//cpuWtsC = 10;
				//gpuWtsC = 10;
				// hit test
				if (cpuMinBounds.contains(mouse))
				{
					// mouse is on sprite!
					std::cout << "hit cpu minus" << std::endl;
					if(cpuWtsC != 1)
						cpuWtsC--;
				}
				else if(cpuPlusBounds.contains(mouse)){
					std::cout << "hit cpu plus" << std::endl;
					cpuWtsC++;
				}
				else if(gpuMinBounds.contains(mouse)){
					std::cout << "hit gpu minus" << std::endl;
					if(gpuWtsC != 1)
						gpuWtsC--;
				}
				else if(gpuPlusBounds.contains(mouse)){
					std::cout << "hit gpu plus" << std::endl;
					gpuWtsC++;
				}
				
			}
			//menu switch
			switch (event.type){
				case sf::Event::KeyReleased:
					switch (event.key.code){
						//up arrow pressed
						case sf::Keyboard::Up:
							if(option == 0){
								menu.MoveUp();		//main menu move up
								if(mainMenuCounter != 1){
									mainMenuCounter--;
								}
							}
							else if(option != 1){
								if(algoOption == 0){
									algoMenu.MoveUp();	//algo menu move up
									if(algoMenuCounter != 1){
										algoMenuCounter--;
									}
								}
								else{
									resMenu.MoveUp();	//result menu move up
									if(resMenuCounter != 1){
										resMenuCounter--;
									}
								}
							}
							break;
						//down arrow pressed
						case sf::Keyboard::Down:
							if(option == 0){
								menu.MoveDown();	//main menu move down
								if(mainMenuCounter != 4){
									mainMenuCounter++;	//counter will not go higher that 4
								}
							}
							else if(option != 1){
								if(algoOption == 0){
									algoMenu.MoveDown();	//algo menu move down
									if(algoMenuCounter != 4){
										algoMenuCounter++;
									}
								}
								else{
									resMenu.MoveDown();		//result menu move down
									if(resMenuCounter != 4){
										resMenuCounter++;
									}
								}
							}
							break;
						case sf::Keyboard::BackSpace:
							if(algoOption != 0){
								if(resOption != 0)
									resOption = 0;
								else
									algoOption = 0;
							}
							else{
								testsRunning = false;
								option = 0;
							}
							break;
						case sf::Keyboard::S:
							if(option == 1){
								testsRunning = true;
								runTests(cpuWtsC, gpuWtsC);
							}
						//return pressed
						case sf::Keyboard::Return:
							//main menu
							if(option == 0){
								switch (menu.GetPressedItem()){
									case 0:
										//Run Tests
										option = 1;
										std::cout << "Your own testing" << std::endl;
										break;
									case 1:
										//View results
										option = 2;
										std::cout << "Your Results" << std::endl;
										break;
									case 2:
										//View my results
										option = 3;
										std::cout << "My results" << std::endl;
										break;
									case 3:
										window.close();
										break;
								}
								break;
							}
							//sub menu for each of the algorithms
							else if(option != 1){
								if(algoOption == 0){
									switch (algoMenu.GetPressedItem()){
										case 0:
											//A* Search
											algoOption = 1;
											std::cout << "A* Search" << std::endl;
											break;
										case 1:
											//Finite State Machines
											algoOption = 2;
											std::cout << "Finite State Machines" << std::endl;
											break;
										case 2:
											//Decision Trees
											algoOption = 3;
											std::cout << "Decision Trees" << std::endl;
											break;
										case 3:
											option = 0;
											algoOption = 0;
											break;
									}
									break;
								}
								else{
									//result menu
									switch (resMenu.GetPressedItem()){
										case 0:
											//Execution time
											resOption = 1;
											std::cout << "Execution time" << std::endl;
											break;
										case 1:
											//Speed per Watt
											resOption = 2;
											std::cout << "Speed per Watt" << std::endl;
											break;
										case 2:
											//Utilization
											resOption = 3;
											std::cout << "Utilization" << std::endl;
											break;
										case 3:
											resOption = 0;
											algoOption = 0;
											break;
									}
									break;
								}
							} 
						}
					break;
			case sf::Event::Closed:
				window.close();
				break;
			}
		}
         
		//prepare frame
        window.clear();

		if(option == 0){
			menu.draw(window);
			//draw text boxes with messages for each option in main menu
			if(mainMenuCounter == 1){
				displayText.setString(runTestsMsg);
				window.draw(displayText);
			}
			else if(mainMenuCounter == 2){
				displayText.setString(viewResultsMsg);
				window.draw(displayText);
			}
			else if(mainMenuCounter == 3){
				displayText.setString(viewMyResultsMsg);
				window.draw(displayText);
			}
		}
		else if(option == 1){
			if(!testsRunning){
				title.setString("Press 'S' to start testing");
				std::string sCpuWtsC = std::to_string(cpuWtsC);
				std::string sGpuWtsC = std::to_string(gpuWtsC);

				cpuWtsCText.setString(sCpuWtsC);
				gpuWtsCText.setString(sGpuWtsC);

				window.draw(cpuMinus);
				window.draw(cpuPlus);
				window.draw(gpuMinus);
				window.draw(gpuPlus);
				window.draw(cpuWattsText);
				window.draw(gpuWattsText);
				window.draw(cpuWtsCText);
				window.draw(gpuWtsCText);
			}
			else
				title.setString("Tests running!");
			window.draw(title);
		}
		//to view users results
		else if(option == 2){
			if(algoOption == 0){
				algoMenu.draw(window);
				//draw text boxes with messages for each option in algo menu
				if(algoMenuCounter == 1){
					displayText.setString(aStarMsg);
					window.draw(displayText);
				}
				else if(algoMenuCounter == 2){
					displayText.setString(fsmMsg);
					window.draw(displayText);
				}
				else if(algoMenuCounter == 3){
					displayText.setString(dtMsg);
					window.draw(displayText);
				}
			}
			else{
				if(resOption == 0)	{
					resMenu.draw(window);
					//draw text boxes with messages for each option in algo menu
					if(resMenuCounter == 1){
						displayText.setString(executionTimeMSg);
						window.draw(displayText);
					}
					else if(resMenuCounter == 2){
						displayText.setString(speedPerWattMsg);
						window.draw(displayText);
					}
					else if(resMenuCounter == 3){
						displayText.setString(utilizationMsg);
						window.draw(displayText);
					}
				}
				else if (resOption == 1){
					title.setString("Execution time");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
				}
				else if (resOption == 2){
					title.setString("Speed up per watt");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
				}
				else if (resOption == 3){
					title.setString("GPU Utilization");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
				}
			}
		}
		//to view my results
		else if(option == 3){
			if(algoOption == 0){
				algoMenu.draw(window);
				//draw text boxes with messages for each option in algo menu
				if(algoMenuCounter == 1){
					displayText.setString(aStarMsg);
					window.draw(displayText);
				}
				else if(algoMenuCounter == 2){
					displayText.setString(fsmMsg);
					window.draw(displayText);
				}
				else if(algoMenuCounter == 3){
					displayText.setString(dtMsg);
					window.draw(displayText);
				}
			}
			//MY A* RESULTS
			else if(algoOption == 1){
				if(resOption == 0){
					resMenu.draw(window);
					//draw text boxes with messages for each option in res menu
					if(resMenuCounter == 1){
						displayText.setString(executionTimeMSg);
						window.draw(displayText);
					}
					else if(resMenuCounter == 2){
						displayText.setString(speedPerWattMsg);
						window.draw(displayText);
					}
					else if(resMenuCounter == 3){
						displayText.setString(utilizationMsg);
						window.draw(displayText);
					}
				}
				else if (resOption == 1){
					title.setString("My A* Execution time");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
					
					//cpu & cuda results
					float* cpuResults = readCSV("CPUastarExecutionTime.csv");
					float* cudaResults = readCSV("astarExecutionTime.csv");

					//check if results exist
					if(cpuResults[0] == -1){
						displayText.setString("Error: Results for CPU A* Execution Time Do Not Exist.\n File must have been deleted");
						window.draw(displayText);
					}
					else{
						cpuPoints = drawData(xAxis, cpuResults, 1);
						for(int i = 0; i < 6; i++)
							window.draw(cpuPoints[i]);
					}

					if(cudaResults[0] == -1){
						displayText.setString("Error: Results for CUDA A* Execution Time Do Not Exist.\nFile must have been deleted");
						window.draw(displayText);
					}
					else{
						cudaPoints = drawData(xAxis, cudaResults, 2);
						for(int i = 0; i < 6; i++)
							window.draw(cudaPoints[i]);
					}

					window.draw(line1);
					window.draw(line2);
					window.draw(line3);
					window.draw(line4);
					window.draw(line5);
					window.draw(line6);
					window.draw(line7);
					window.draw(line8);
					window.draw(xline1);
					window.draw(xline2);
					window.draw(xline3);
					window.draw(xline4);
					window.draw(xline5);
					window.draw(xline6);
				}
				else if (resOption == 2){
					title.setString("My A* Speed up per watt");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
				}
				else if (resOption == 3){
					title.setString("My A* GPU Utilization");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
				}
			}
			//MY FSM RESULTS
			else if(algoOption == 2){
				if(resOption == 0){
					resMenu.draw(window);
					//draw text boxes with messages for each option in res menu
					if(resMenuCounter == 1){
						displayText.setString(executionTimeMSg);
						window.draw(displayText);
					}
					else if(resMenuCounter == 2){
						displayText.setString(speedPerWattMsg);
						window.draw(displayText);
					}
					else if(resMenuCounter == 3){
						displayText.setString(utilizationMsg);
						window.draw(displayText);
					}
				}
				else if (resOption == 1){
					title.setString("My FSM Execution time");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
					
					//cpu & cuda results
					float* cpuResults = readCSV("CPUfsmExecutionTime.csv");
					float* cudaResults = readCSV("fsmExecutionTime.csv");
					cpuPoints = drawData(xAxis, cpuResults, 1);
					cudaPoints = drawData(xAxis, cudaResults, 2);

					for(int i = 0; i < 6; i++){
						window.draw(cpuPoints[i]);
						window.draw(cudaPoints[i]);
					}
				}
				else if (resOption == 2){
					title.setString("My FSM Speed up per watt");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
				}
				else if (resOption == 3){
					title.setString("My FSM GPU Utilization");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
				}
			}
			//MY DT RESULTS
			else if(algoOption == 3){
				if(resOption == 0){
					resMenu.draw(window);
					//draw text boxes with messages for each option in res menu
					if(resMenuCounter == 1){
						displayText.setString(executionTimeMSg);
						window.draw(displayText);
					}
					else if(resMenuCounter == 2){
						displayText.setString(speedPerWattMsg);
						window.draw(displayText);
					}
					else if(resMenuCounter == 3){
						displayText.setString(utilizationMsg);
						window.draw(displayText);
					}
				}
				else if (resOption == 1){
					title.setString("My DT Execution time");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
					
					//cpu & cuda results
					float* cpuResults = readCSV("CPUdtExecutionTime.csv");
					float* cudaResults = readCSV("dtExecutionTime.csv");
					cpuPoints = drawData(xAxis, cpuResults, 1);
					cudaPoints = drawData(xAxis, cudaResults, 2);

					for(int i = 0; i < 6; i++){
						window.draw(cpuPoints[i]);
						window.draw(cudaPoints[i]);
					}
				}
				else if (resOption == 2){
					title.setString("My DT Speed up per watt");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
				}
				else if (resOption == 3){
					title.setString("My DT GPU Utilization");
					window.draw(title);
					window.draw(time);
					window.draw(agents);
					window.draw(xAxis);
					window.draw(yAxis);
				}
			}
		}
        // Finally, display rendered frame on screen 
        window.display(); 
    } //loop back for next frame
   
    return EXIT_SUCCESS; 
}
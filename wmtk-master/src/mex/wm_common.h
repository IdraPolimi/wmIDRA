#ifndef __WM_COMMON__
#define __WM_COMMON__

#include "WMtk.h"

#define MAX_WMS_NUMBER 50
#define MAX_FEATURES_NUMBER 8192
#define MAX_CHUNKS_NUMBER 100


typedef struct {
	int index;
	double features[MAX_FEATURES_NUMBER];
    int wm_size, state_size, chunk_size;
    double current_reward;
    list<Chunk> * current_chunks;
} wm_state;


typedef struct {
    double features[MAX_FEATURES_NUMBER];
} chunk_data;



typedef struct {
    WorkingMemory * wm_modules[MAX_WMS_NUMBER];
    int count_wm_modules;
}WM;

bool checkIndex(WM&, int index);

int newWMModule(WM&, int wm_size,  int state_size, int chunk_size);

bool setState(WM&, int index, double state[]);
double setReward(WM&, int index, double reward);

int addChunk(WM&, int index, double chunk_data[]);
int countChunks(WM&, int index);
int clearChunks(WM&, int index);


void initWM(WM&);
int tickWM(WM&, int index);

int countRetainedChunks(WM&, int index);
Chunk getRetainedChunk(WM&, int index, int chunkNo);

double getTDError(WM&, int index);

bool setExplorationPercentage(WM&, int index, double val);

bool newEpisode(WM&, int index);

double setGamma(WM&, int index, double value);

double setLearningRate(WM&, int index, double value);

wm_state * GetStateDataStructure(WorkingMemory* wm);

//----- used by WorkingMemory
double reward_function(WorkingMemory&);
void state_function(FeatureVector&, WorkingMemory&);
void chunk_function(FeatureVector&, Chunk&, WorkingMemory&);
void delete_function(Chunk&);
//-----

void printArray(double * data, int length, const char * message);

#endif
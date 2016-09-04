#ifndef __WM_COMMON__
#define __WM_COMMON__

#include "WMtk.h"

#define MAX_WMS_NUMBER 50
#define MAX_FEATURES_NUMBER 100
#define MAX_CHUNKS_NUMBER 100

#define OR_ONLY

typedef struct {
    double features[MAX_FEATURES_NUMBER];
    int features_count;
} chunk_data;

typedef struct {
    double features[MAX_FEATURES_NUMBER];
    list<Chunk> current_chunks;
    int wm_size, state_size, chunk_size;
} state_data_structure;


typedef struct {
    double current_reward[MAX_WMS_NUMBER];
    state_data_structure current_state[MAX_WMS_NUMBER];
    WorkingMemory * wm_modules[MAX_WMS_NUMBER];
    int count_wm_modules;
}WM;

extern WM wm_framework;

bool checkIndex(WM&, int index);

int newWMModule(WM&, int wm_size,  int state_size, int chunk_size);
void initWM(WM&);
int tickWM(WM&, int index);

bool setState(WM&, int index, double state[], int length);
double setReward(WM&, int index, double reward);

int addChunk(WM&, int index, double chunk_data[], int length);

int countRetainedChunks(WM&, int index);
Chunk getRetainedChunk(WM&, int index, int chunkNo);

int countCandidateChunks(WM&, int index);
bool clearCandidateChunks(WM&, int index);

double getTDError(WM&, int index);

bool setExplorationPercentage(WM&, int index, double val);

bool newEpisode(WM& wmf, int index);

double setGamma(WM& wmf, int index, double value);



//----- used by WorkingMemory
double reward_function(WorkingMemory&);
void state_function(FeatureVector&, WorkingMemory&);
void chunk_function(FeatureVector&, Chunk&, WorkingMemory&);
void delete_function(Chunk&);
//-----


#endif
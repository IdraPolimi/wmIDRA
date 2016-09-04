#include "wm_common.h"
#include "mex.h"

bool checkIndex(WM&wmf, int index)
{
    return index >= 0 && index < MAX_WMS_NUMBER;
}

double reward_function(WorkingMemory& wm)
{
    wm_state* state = GetStateDataStructure(&wm);
    return state -> current_reward;
}


void state_function(FeatureVector& v, WorkingMemory& wm)
{
    wm_state* state = GetStateDataStructure(&wm);

    double *features = state -> features; 
    
    for(int i = 0; i < state -> state_size; i++)
        v.setValue(i, features[i]);
    

//     cout << "state 1\n";
//     for(int i = 0; i < state -> state_size; i++)
//         cout << v.getValue(i) << "  ";
//     cout <<"\n";
    
}

void chunk_function(FeatureVector& v, Chunk& c, WorkingMemory& wm)
{
    chunk_data * cd = (chunk_data*) c.getData();
    wm_state* sds = GetStateDataStructure(&wm);
    
    for(int i = 0; i < sds -> chunk_size; i++)
        v.setValue(i, cd -> features[i]);
    
//      cout << "chunk\n";
//      for(int i = 0; i < sds -> chunk_size; i++)
//          cout << v.getValue(i) << "  ";
//      cout <<"\n";
}

void delete_function(Chunk& c)
{
    free(c.getData());
}

int newWMModule(WM &wmf, int wm_size, int state_size, int chunk_size)
{
    int index = wmf.count_wm_modules;
    
    if(!checkIndex(wmf, index))
        return -1;
    
    wm_state * state = (wm_state*) malloc( sizeof(wm_state) );

    state -> index = index;
    state -> wm_size = wm_size;
    state -> state_size = state_size;
    state -> chunk_size = chunk_size;
    
    state -> current_chunks = new list<Chunk>();

    wmf.wm_modules[index] = new WorkingMemory(wm_size,
                                       state_size,
                                       chunk_size,
                                       state,
                                       &reward_function,
                                       &state_function,
                                       &chunk_function,
                                       &delete_function,
                                       false,
                                       NO_OR
                                       );

    wmf.count_wm_modules++;
    return index;
}


wm_state* GetStateDataStructure(WorkingMemory* wm)
{
    return (wm_state*) wm -> getStateDataStructure();
}

bool setState(WM& wmf, int index, double state[])
{
    if(checkIndex(wmf, index))
    {
        wm_state* sds = GetStateDataStructure( wmf.wm_modules[index] );
        
//         double diff[sds -> state_size];
//         for (int i = 0; i < sds -> state_size; i++)
//             diff[i] = (sds -> features[i]) - state[i];
//         printArray(diff, sds -> state_size, "ss");
        
        for (int i = 0; i < sds -> state_size; i++)
            sds -> features[i] = state[i];
        
        
        return true;
    }
    return false;
}

double setReward(WM& wmf, int index, double reward)
{
    wm_state* sds = GetStateDataStructure( wmf.wm_modules[index] );
    if(checkIndex(wmf, index))
    {
        sds -> current_reward = reward;
    }
    return sds -> current_reward;
}

int addChunk(WM& wmf, int index, double features[])
{
    if(!checkIndex(wmf, index))
        return -1;
    
    wm_state* sds = GetStateDataStructure( wmf.wm_modules[index] );
    
    int count = sds -> current_chunks -> size();
    
    if(count >= MAX_CHUNKS_NUMBER)
        return MAX_CHUNKS_NUMBER;
    
    chunk_data * cd = (chunk_data*)malloc(sizeof(chunk_data));
    
    
    for(int i = 0; i < sds -> chunk_size; i++)
        cd -> features[i] = features[i];
    
    Chunk cc(cd, string("chunk"));
    
    
    
    sds -> current_chunks -> push_back( cc );
    
    
    cd = (chunk_data*)cc.getData();
    
//     printArray(((chunk_data*)cc.getData()) -> features, sds -> chunk_size, "CHUNK");
//     printArray(features, sds -> chunk_size, "addChunk() in features");
//     printArray(cd -> features, sds -> chunk_size, "chunk_data features");

    return sds -> current_chunks -> size();
}

int countChunks(WM& wmf, int index)
{
    wm_state* sds = GetStateDataStructure( wmf.wm_modules[index] );
    return sds -> current_chunks -> size();
}

int clearChunks(WM& wmf, int index)
{
    if(checkIndex(wmf, index))
    {
        wm_state* sds = GetStateDataStructure( wmf.wm_modules[index] );
        sds -> current_chunks -> clear();
    }
    return countChunks(wmf, index);

}

void initWM(WM& wmf)
{
    for(int i = 0; i < wmf.count_wm_modules; i++)
    {
        wm_state* sds = GetStateDataStructure( wmf.wm_modules[i] );
        sds -> current_chunks -> clear();
        delete sds -> current_chunks;
        delete wmf.wm_modules[i];
    }
    wmf.count_wm_modules = 0;
}

int tickWM(WM &wmf, int index)
{
    if(!checkIndex(wmf, index))
        return -1;

    wm_state* sds = GetStateDataStructure( wmf.wm_modules[index] );

    list<Chunk> cl = *(sds -> current_chunks);
    

    int res = wmf.wm_modules[index] -> tickEpisodeClock(cl, true);
    clearChunks(wmf, index);
    setReward(wmf, index, 0);
    return res;
}


int countRetainedChunks(WM& wmf, int index)
{
    if(!checkIndex(wmf, index))
        return -1;
    
    return wmf.wm_modules[index] -> getNumberOfChunks();
}

Chunk getRetainedChunk(WM& wmf, int index, int chunkNo)
{
    Chunk c(NULL,"ERROR");
    if(!checkIndex(wmf, index))
        return c;
    
//     wm_state* sds = GetStateDataStructure( wmf.wm_modules[index] );
//     printArray(((chunk_data*)(wmf.wm_modules[index] -> getChunk(chunkNo)).getData()) -> features, sds -> chunk_size, "CHUNK");
    
    return wmf.wm_modules[index] -> getChunk(chunkNo);
}


double getTDError(WM& wmf, int index)
{
    if(!checkIndex(wmf, index))
        return -1;
    
    WorkingMemory * wm = wmf.wm_modules[index];
    
    return (wm -> getCriticNetwork() -> getCriticLayer()) -> getTDError(0);
    
}

bool setExplorationPercentage(WM& wmf, int index, double val)
{
    if(!checkIndex(wmf, index))
        return -1;
    WorkingMemory * wm = wmf.wm_modules[index];
    return wm -> setExplorationPercentage(val);
}

bool newEpisode(WM& wmf, int index)
{
    WorkingMemory * wm = wmf.wm_modules[index];
    return wm -> newEpisode(true);
}


double setGamma(WM& wmf, int index, double value)
{
    if(!checkIndex(wmf, index))
        return -1;
    
    WorkingMemory * wm = wmf.wm_modules[index];
    return wm -> getCriticNetwork() -> setGamma(value);
}

double setLearningRate(WM& wmf, int index, double value)
{
    if(!checkIndex(wmf, index))
        return -1;
    WorkingMemory * wm = wmf.wm_modules[index];
    return wm -> getCriticNetwork() -> setLearningRate(value);
}

void printArray(double * data, int length, const char* message)
{
    mexPrintf("%s\n", message);
    for(int i = 0; i < length; i++)
        cout << data[i] << "  ";
    
    mexPrintf("\n---\n");
}
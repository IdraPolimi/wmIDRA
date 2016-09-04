#include "wm_common.h"
#include "mex.h"

WM wm_framework;

bool checkIndex(WM&wm_framework, int index)
{
    return index >= 0 && index < MAX_WMS_NUMBER;
}


double reward_function(WorkingMemory& wm)
{
    return wm_framework.current_reward[wm.index];
}


void state_function(FeatureVector& v, WorkingMemory& wm)
{
    double *features = wm_framework.current_state[wm.index].features;
    for(int i = 0; i < v.getSize(); i++)
        v.setValue(i, features[i]);
}

void chunk_function(FeatureVector& v, Chunk& c, WorkingMemory& wm)
{
    chunk_data * cd = (chunk_data*) c.getData();
    
    for(int i = 0; i < min(v.getSize(), cd -> features_count); i++)
        v.setValue(i, cd -> features[i]);
    
}

void delete_function(Chunk& c)
{
    free( c.getData() );
}


int newWMModule(WM &wmf, int wm_size, int state_size, int chunk_size)
{
    int index = wmf.count_wm_modules;
    
    if(!checkIndex(wmf, index))
        return -1;
    
    wmf.current_state[index].wm_size = wm_size;
    wmf.current_state[index].state_size = state_size;
    wmf.current_state[index].chunk_size = chunk_size;
    
     wmf.wm_modules[index] = new WorkingMemory(wm_size,
                                       state_size,
                                       chunk_size,
                                       NULL,
                                       &reward_function,
                                       &state_function,
                                       &chunk_function,
                                       &delete_function,
                                       false,
                                       NO_OR
                                       );
    wmf.wm_modules[index] -> index = index;
    wmf.count_wm_modules++;
    return index;
}

bool setState(WM& wm_framework, int index, double state[], int length)
{
    if(!checkIndex(wm_framework, index)) return false;
    
    length = min(length, wm_framework.current_state[index].state_size);
    
    for (int i = 0; i < length; i++)
        wm_framework.current_state[index].features[i] = state[i];
    return true;
}

double setReward(WM& wm_framework, int index, double reward)
{
    if(checkIndex(wm_framework, index))
    {
        wm_framework.current_reward[index] = reward;
        return wm_framework.current_reward[index];
    }
    return -1;
}

int addChunk(WM& wmf, int index, double features[], int length)
{
    if(!checkIndex(wmf, index))
        return -1;
    
    
    int count = wmf.current_state[index].current_chunks.size();
    
    if(count >= MAX_CHUNKS_NUMBER)
        return MAX_CHUNKS_NUMBER;
    
    chunk_data * cd = (chunk_data*)malloc(sizeof(chunk_data));
    
    int max_size = wmf.current_state[index].chunk_size;
    
     for(int i = 0; i < min(max_size, length); i++)
         cd -> features[i] = features[i];
    
    cd -> features_count = min(max_size, length);
    
    
    wmf.current_state[index].current_chunks.push_back( Chunk(cd,string("data")) );
    
    return wmf.current_state[index].current_chunks.size();
}

void initWM(WM& wmf)
{
    for(int i = 0; i < wmf.count_wm_modules; i++)
    {
        wmf.current_state[i].current_chunks.clear();
        delete wmf.wm_modules[i];
    }
    wmf.count_wm_modules = 0;
}

int tickWM(WM &wmf, int index)
{
    if(!checkIndex(wmf, index))
        return -1;
    int res = wmf.wm_modules[index] -> tickEpisodeClock(wmf.current_state[index].current_chunks);
    wmf.current_state[index].current_chunks.clear();
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
    Chunk c(NULL, "null");
    if(!checkIndex(wmf, index))
        return c;
    
    
    c = wmf.wm_modules[index] -> getChunk(chunkNo);
    return c;
}


int countCandidateChunks(WM& wmf, int index)
{
    return wmf.current_state[index].current_chunks.size();
}

bool clearCandidateChunks(WM& wmf, int index)
{
    if(checkIndex(wmf, index))
    {
        wmf.current_state[index].current_chunks.clear();
        return true;
    }
    return false;

}

double getTDError(WM& wmf, int index)
{
    if(!checkIndex(wmf, index))
        return -1;
    
    WorkingMemory * wm = wmf.wm_modules[index];
    double reward = wmf.current_reward[wm -> index];
    
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

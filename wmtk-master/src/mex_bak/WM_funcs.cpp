#include "wm_common.h"
#include "mex.h"

extern WM wm_framework;
#define INIT 1          //()
#define NEW_MODULE 2    //index = (wm_size, state_size, chunk_size)
#define ADD_CHUNK 3     //ok    = (index, chunk_features)
#define SET_STATE 4     //ok    = (index, state_features)
#define SET_REWARD 5    //(index, reward)
#define EPISODE_TICK 6    //(index)
#define COUNT_RETEINED_CHUNKS 7 //(index)
#define GET_RETAINED_CHUNK 8    //(index, chunkNo)
#define GET_TD_ERROR 9          //(index)
#define SET_EXPLORATION_PERC 10 //(index, percentage)
#define NEW_EPISODE 11 //(index)
#define SET_GAMMA 12 //(index, value)
#define CLEAR_CHUNKS 13 //(index)
#define COUNT_CHUNKS 14 //(index)

void checkArgs(int args_count, int min_args_count)
{
    if(args_count < min_args_count)
        mexErrMsgTxt("Wrong number of input arguments.");
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    checkArgs(nrhs, 1);
    
    int op =  (int)mxGetScalar(prhs[0]);
    
    switch(op)
    {
        case INIT: 
            initWM(wm_framework);
            break;
            
        case NEW_MODULE: 
        {
            checkArgs(nrhs, 4);
            
            int wm_size = (int)mxGetScalar(prhs[1]);
            int state_size = (int)mxGetScalar(prhs[2]);
            int chunk_size = (int)mxGetScalar(prhs[3]);
            int res = newWMModule(wm_framework, wm_size, state_size, chunk_size);

            if(nlhs > 0)
                plhs[0] = mxCreateDoubleScalar(res);
            break;
        }    
        case ADD_CHUNK:
        {
            checkArgs(nrhs, 3);

            int M = mxGetM(prhs[2]);
            int N = mxGetN(prhs[2]);

            int length = N*M;

            int index = (int)mxGetScalar(prhs[1]);
            double *state = mxGetPr(prhs[2]);

            int res =  addChunk(wm_framework, index, state, length);

            if(nlhs > 0)
                plhs[0] = mxCreateDoubleScalar(res);
            break;
        }    
        case SET_STATE: 
        {
            checkArgs(nrhs, 3);

            int M = mxGetM(prhs[2]);
            int N = mxGetN(prhs[2]);

            int length = N*M;

            int index =        (int)mxGetScalar(prhs[1]);
            double *state = mxGetPr(prhs[2]);


            bool res = setState(wm_framework, index, state, length);

            if(nlhs > 0)
                plhs[0] = mxCreateDoubleScalar(res);
            break;
        }
        
        case SET_REWARD:
        {
            checkArgs(nrhs,3);
            int index =        (int)mxGetScalar(prhs[1]);
            double reward = (double)mxGetScalar(prhs[2]);

            double res = setReward(wm_framework, index, reward);
            if(nlhs > 0)
                plhs[0] = mxCreateDoubleScalar(res);
            break;
        }
        
        case EPISODE_TICK:
        {
            checkArgs(nrhs,2);
            int index =        (int)mxGetScalar(prhs[1]);
            int res = tickWM(wm_framework, index);
            plhs[0] = mxCreateDoubleScalar(res);
            break;
        }
        
        case COUNT_RETEINED_CHUNKS:
        {
            
            checkArgs(nrhs,2);
            int index =        (int)mxGetScalar(prhs[1]);
            int res = countRetainedChunks(wm_framework, index);
            plhs[0] = mxCreateDoubleScalar(res);
            break;
        }
        
        case GET_RETAINED_CHUNK:
        {
            checkArgs(nrhs,3);
            int index =        (int)mxGetScalar(prhs[1]);
            int chunkNo =        (int)mxGetScalar(prhs[2]);
            Chunk c = getRetainedChunk(wm_framework, index, chunkNo);
            chunk_data * cd = (chunk_data*)c.getData();
            
            if(cd == NULL)
            {
                plhs[0] = mxCreateDoubleScalar(-1);
            }
            else
            {
                plhs[0] = mxCreateDoubleMatrix(1, cd -> features_count, mxREAL);
                double * res = mxGetPr(plhs[0]);
                for(int i = 0; i < cd -> features_count; i++)
                    res[i] = cd -> features[i];
            }
            
            break;
        }
        
        case GET_TD_ERROR:
        {
            checkArgs(nrhs,2);
            int index =        (int)mxGetScalar(prhs[1]);
            
            
            
            plhs[0] = mxCreateDoubleScalar(getTDError(wm_framework, index));
            
            
            break;
        }
        
        case SET_EXPLORATION_PERC:
        {
            checkArgs(nrhs,3);
            int index =        (int)mxGetScalar(prhs[1]);
            double perc  =        (double)mxGetScalar(prhs[2]);
            
            bool res = setExplorationPercentage(wm_framework, index, perc);
            if(nlhs > 0)
                plhs[0] = mxCreateDoubleScalar(res);
            break;
        }
        
        case NEW_EPISODE:
        {
            checkArgs(nrhs,2);
            int index =        (int)mxGetScalar(prhs[1]);
            
            bool res = newEpisode(wm_framework, index);
            if(nlhs > 0)
                plhs[0] = mxCreateDoubleScalar(res);
            break;
        }
        
        case SET_GAMMA:
        {
            checkArgs(nrhs,3);
            int index =        (int)mxGetScalar(prhs[1]);
            double value  =        (double)mxGetScalar(prhs[2]);
            
            double res = setGamma(wm_framework, index, value);
            if(nlhs > 0)
                plhs[0] = mxCreateDoubleScalar(res);
            break;
        }
        
        case CLEAR_CHUNKS:
        {
            checkArgs(nrhs,2);
            int index =        (int)mxGetScalar(prhs[1]);
            
            bool res = clearCandidateChunks(wm_framework, index);
            if(nlhs > 0)
                plhs[0] = mxCreateDoubleScalar(res);
            break;
        }
        
        case COUNT_CHUNKS:
        {
            checkArgs(nrhs,2);
            int index =        (int)mxGetScalar(prhs[1]);
            
            int res = countCandidateChunks(wm_framework, index);
            if(nlhs > 0)
                plhs[0] = mxCreateDoubleScalar(res);
            break;
        }
    }
    
}
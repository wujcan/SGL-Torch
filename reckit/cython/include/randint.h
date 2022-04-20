/*
@author: Zhongchuan Sun
*/
#ifndef RANDINT_H
#define RANDINT_H

#include "thread_pool.h"
#include <vector>
#include <random>
#include <unordered_set>
#include <future>
using std::unordered_set;
using std::future;
using std::vector;


typedef unordered_set<int> int_set;

std::mt19937 _gen(2020);

template <typename T>
void _random_int(T &sampler, int size, bool replace, const int_set* exclusion, int* result)
{
    if(replace)
    {// replace is True
        if(!exclusion)
        {// if exclusion is NULL
            int i = 0;
            while(i<size)
            {
                result[i] = sampler(_gen);
                i++;
            }
        }
        else
        {// if exclusion is not NULL
            int s = -1;
            int i = 0;
            while(i<size)
            {
                s = sampler(_gen);
                if(!exclusion->count(s))
                {// s not in exclusion
                    result[i] = s;
                    i++;
                }
            }
        }
    }
    else
    {// replace is False
        int_set _exclusion;
        if(exclusion)
        {// _exclusion contained exclusion and sampled integer
            _exclusion.insert(exclusion->begin(), exclusion->end());
        }
        // int_set _exclusion(exclusion);
        int s = -1;
        int i = 0;
        while(i<size)
        {
            s = sampler(_gen);
            if(!_exclusion.count(s))
            {// s not in exclusion
                result[i] = s;
                i++;
                _exclusion.insert(s);
            }
        }
    }
}


int c_randint_choice(int high, int size, bool replace, const float* prob, const int_set* exclusion, int* result)
{//sample from [0,high)
    if(prob)
    {// sample from [0,high) according to the given prob
        std::discrete_distribution<> sampler(prob, prob+high);
        _random_int(sampler, size, replace, exclusion, result);
    }
    else
    {// sample from [0,high) uniformly
        std::uniform_int_distribution<> sampler(0, high-1);
        _random_int(sampler, size, replace, exclusion, result);
    }
    return 0;
}

int c_batch_randint_choice(int high, const int* size_ptr, int batch_num, bool replace,
                           const float* prob, const int_set* exclusion, int n_threads, int* result)
{
    const float* p_ptr = 0;
    int* result_ptr = result;
    int size = 0;
    const int_set* exc_ptr = 0;
    if(n_threads > 1)
    {// multi-thread
        ThreadPool pool(n_threads);
        vector< future< int > > sync_results;

        for(int i=0; i<batch_num; i++)
        {
            size = size_ptr[i];
            p_ptr = prob?(prob+i*high):0;
            exc_ptr = exclusion?exclusion+i:0;
            sync_results.emplace_back(pool.enqueue(c_randint_choice, high, size, replace, p_ptr, exc_ptr, result_ptr));
            result_ptr = result_ptr + size;
        }

        for(auto && result: sync_results)
        {
            result.get();  // join
        }
    }
    else
    {//single-thread
        for(int i=0; i<batch_num; i++)
        {
            size = size_ptr[i];
            p_ptr = prob?(prob+i*high):0;
            exc_ptr = exclusion?exclusion+i:0;
            c_randint_choice(high, size, replace, p_ptr, exc_ptr, result_ptr);
            result_ptr = result_ptr + size;
        }
    }
    return 0;
}

#endif

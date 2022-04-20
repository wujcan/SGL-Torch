/*
@author: Zhongchuan Sun
*/
#ifndef SORT_H
#define SORT_H

#include "thread_pool.h"
#include <vector>
#include <algorithm>
#include <numeric>  // std::iota
#include <future>
using std::vector;
using std::future;


//**********_T_sort**********//
template <typename Tv, typename Tr>
int _T_sort_2d(int(*func)(const Tv*, int, Tr*, bool), const Tv *matrix_ptr,
               int n_rows, int n_cols, int n_threads, Tr *results_ptr, bool reverse=false)
{
    auto array_ptr = matrix_ptr;
    auto r_ptr = results_ptr;
    if(n_threads>1)
    {
        ThreadPool pool(n_threads);
        vector< future< int > > sync_results;

        for(int i=0; i<n_rows; ++i)
        {
            array_ptr = matrix_ptr + i*n_cols;
            r_ptr = results_ptr + i*n_cols;
            sync_results.emplace_back(pool.enqueue(func, array_ptr, n_cols, r_ptr, reverse));
        }

        for(auto && result: sync_results)
        {
            result.get();  // join
        }
    }
    else
    {
        for(int i=0; i<n_rows; ++i)
        {
            array_ptr = matrix_ptr + i*n_cols;
            r_ptr = results_ptr + i*n_cols;
            func(array_ptr, n_cols, r_ptr, reverse);
        }
    }

    return 0;
}


//**********sort**********//
template <typename T>
int c_sort_1d(const T *array_ptr, int length, T *result, bool reverse=false)
{
    std::copy(array_ptr, array_ptr+length, result);
    std::sort(result, result+length);
    if(reverse)
    {
        std::reverse(result, result+length);
    }
    return 0;
}


template <typename T>
int c_sort_2d(const T *matrix_ptr, int n_rows, int n_cols, int n_threads, T *results_ptr, bool reverse=false)
{
    return _T_sort_2d(c_sort_1d, matrix_ptr, n_rows, n_cols, n_threads, results_ptr, reverse);
}


//**********arg_sort**********//
template <typename T>
int c_arg_sort_1d(const T *array_ptr, int length, int *result, bool reverse=false)
{
    std::iota(result, result+length, 0);

    const auto function = [& array_ptr](const int &x1, const int &x2)->bool{return array_ptr[x1]<array_ptr[x2];};
    std::sort(result, result+length, function);
    if(reverse)
    {
        std::reverse(result, result+length);
    }
    return 0;
}


template <typename T>
int c_arg_sort_2d(const T *matrix_ptr, int n_rows, int n_cols, int n_threads, int *results_ptr, bool reverse=false)
{
    return _T_sort_2d(c_arg_sort_1d, matrix_ptr, n_rows, n_cols, n_threads, results_ptr, reverse);
}


//**********_T_top_k**********//
template <typename Tv, typename Tr>
int _T_top_k_2d(int(*func)(const Tv*, int, int, Tr*), const Tv *matrix_ptr,
                int n_rows, int n_cols, int top_k, int n_threads, Tr *results_ptr)
{
    auto array_ptr = matrix_ptr;
    auto r_ptr = results_ptr;
    if(n_threads>1)
    {
        ThreadPool pool(n_threads);
        vector< future< int > > sync_results;

        for(int i=0; i<n_rows; ++i)
        {
            array_ptr = matrix_ptr + i*n_cols;
            r_ptr = results_ptr + i*top_k;
            sync_results.emplace_back(pool.enqueue(func, array_ptr, n_cols, top_k, r_ptr));
        }

        for(auto && result: sync_results)
        {
            result.get();  // join
        }
    }
    else
    {
        for(int i=0; i<n_rows; ++i)
        {
            array_ptr = matrix_ptr + i*n_cols;
            r_ptr = results_ptr + i*top_k;
            func(array_ptr, n_cols, top_k, r_ptr);
        }
    }
    return 0;
}


//**********top_k**********//
template <typename T>
int c_top_k_1d(const T *array_ptr, int length, int top_k, T *result)
{
    const auto function = [](const T &x1, const T &x2)->bool{return x1>x2;};

    std::partial_sort_copy(array_ptr, array_ptr+length, result, result+top_k, function);
    return 0;
}


template <typename T>
int c_top_k_2d(const T *matrix_ptr, int n_rows, int n_cols, int top_k, int n_threads, T *results_ptr)
{
    return _T_top_k_2d(c_top_k_1d, matrix_ptr, n_rows, n_cols, top_k, n_threads, results_ptr);
}


//**********arg_top_k**********//
template <typename T>
int c_arg_top_k_1d(const T *array_ptr, int length, int top_k, int *result)
{
    vector<int> index(length);
    std::iota(index.begin(), index.end(), 0);

    const auto function = [& array_ptr](const int &x1, const int &x2)->bool{return array_ptr[x1]>array_ptr[x2];};
    std::partial_sort_copy(index.begin(), index.end(), result, result+top_k, function);
    return 0;
}


template <typename T>
int c_arg_top_k_2d(const T *matrix_ptr, int n_rows, int n_cols, int top_k, int n_threads, int *results_ptr)
{
    return _T_top_k_2d(c_arg_top_k_1d, matrix_ptr, n_rows, n_cols, top_k, n_threads, results_ptr);
}


#endif

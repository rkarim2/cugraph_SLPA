/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/prims/extract_if_e.cuh>
#include <cugraph/prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh>
#include <cugraph/prims/update_edge_partition_src_dst_property.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <thrust/random.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {
//specfic functors for SLPA
namespace {

__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}


struct get_label 
{
    int * col;
    thrust::tuple<int,int> * mem;
    int * memnnz;
    int n;
    //int * _col, int * _mem, int * _memnnz,
    get_label(int _n, thrust::device_vector<int> _col, thrust::device_vector<thrust::tuple<int,int> > _mem, thrust::device_vector<int> _memnnz) {
        col= thrust::raw_pointer_cast(&_col[0]);
        mem = thrust::raw_pointer_cast(&_mem[0]);
        memnnz= thrust::raw_pointer_cast(&_memnnz[0]);
        //labellist = thrust::raw_pointer_cast(&_labellist[0]);
        n = _n;

    }
  __device__
  thrust::tuple<int,int> operator()(unsigned int thread_id)
  {
    unsigned int seed = hash(thread_id);
    if(thread_id == 0) {
        printf("this is n %d\n this is col[0] %d\n", n, col[0]);
        printf("this is mem[0] %d\n", thrust::get<0>(mem[5]));
        printf("this is memnnz[0] %d\n", memnnz[2]);
    }
    // seed a random number generator
    thrust::default_random_engine rng(seed);

    // create a mapping from random numbers to [0,1)
    thrust::uniform_real_distribution<float> u01(0,1);
    if(thread_id == 0) {
        printf("This is the u01 %f\n", u01(rng));
        printf("this is memnnz col thread %d\n", memnnz[col[thread_id]]);
        printf("THis is the range u01 %f\n", u01(rng)*2.0);
        printf("this is the offset %d\n", (int)round(u01(rng)*memnnz[col[thread_id]]));
    }
      // draw a sample from the unit square
    int offset = (int)truncf(u01(rng)*memnnz[col[thread_id]]);
    
    // printf("%d\n", mem[col[thread_id] * n + offset]);
    // labellist[thread_id] = mem[col[thread_id] * n + offset];
    // printf("%d\n", labellist[thread_id]);
    // return labellist[thread_id];
    thrust::tuple<int, int> val(thrust::get<0>(mem[col[thread_id] * n + offset]), 1);
    return mem[col[thread_id] * n + offset];
  }
};


struct get_frequency 
{
    int * row_id;
    int * row;
    thrust::tuple<int,int> * mem;
    thrust::tuple<int,int> * labellist;
    int n;
    //int * _col, int * _mem, int * _memnnz,
    get_frequency(int _n, thrust::device_vector<int> _row, thrust::device_vector<thrust::tuple<int,int> > _mem, thrust::device_vector<thrust::tuple<int,int> > _labellist, thrust::device_vector<int> _row_id) {
        row= thrust::raw_pointer_cast(&_row[0]);
        mem = thrust::raw_pointer_cast(&_mem[0]);
        labellist = thrust::raw_pointer_cast(&_labellist[0]);
        row_id = thrust::raw_pointer_cast(&_row_id[0]);
        n = _n;
    }
  __device__
  thrust::tuple<int,int> operator()(unsigned int thread_id)
  {
    int offset1 = row[row_id[thread_id]];
    int offset2 = row[row_id[thread_id]+1];
    int count = 0;
    for(int i = offset1; i < offset2; i++) {
      if(thrust::get<0>(labellist[thread_id]) == thrust::get<0>(labellist[i]))
        count++;
    }
    thrust::get<1>(labellist[thread_id]) = count;
    return labellist[thread_id];
  }
};
template <typename edge_t>
struct reduce_per_offset
{
    __device__ void operator()(labellist) const 
    { 
        //get column offsets from graph
        //use those to go over specific labellist range and find frequency
    }
};

}
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void SLPA(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                    std::optional<raft::device_span<vertex_t const>> vertices,
                    raft::device_span<edge_t> counts,
                    bool do_expensive_check)
{
    //set up vectors
    //label list vector of size # edges
    rmm::device_uvector<edge_t> labellist(graph_view.number_of_edges(), handle.get_stream());
    //memory vector of size n by n
    //FIXME needs to be n by outdegree + 1 or T
    rmm::device_uvector<vertex_t> memory(cur_graph_view.number_of_vertices() * cur_graph_view.number_of_vertices(), handle.get_stream());
    //# of valid entries in memory vector for each vertex
    rmm::device_uvector<vertex_t> memnnz(cur_graph_view.number_of_vertices(), handle.get_stream());


    //Kernel 1 propagate labels by edge to all vertices
    //currently waiting on specific primitive to better enable this
    {
        thrust::sequence(labellist.begin(), labellist.end());
    }
    //Kernel 2 get the frequency of labels 
    //assumption label list is full of values
    {
        //go over list and get the freqency 
        //need a custom function to pass to transform
        thrust::transform(labellist.begin(), labellist.end(), labellist.begin(), frequency_per_elem(labellist));
    }

    //Kernel 3 labellist reduction
    {
        thrust::transform(labellist.begin(), labellist.end(), labellist.begin(), reduce_per_offset(labellist));
    }

    //Kernel 4 memmory label update
    {
        
    }

    return;
}
}
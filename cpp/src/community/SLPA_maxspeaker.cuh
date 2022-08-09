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

struct tupleMax
{
    template<typename T, typename T1>
    __host__ __device__
    thrust::tuple<T, T> operator()(thrust::tuple<T, T> t0, thrust::tuple<T1, T1> t1)
    {
            if(thrust::get<1>(t0) < thrust::get<1>(t1)) {
              return t1;
            }
            else 
              return t0;
            //return thrust::make_tuple(thrust::get<0>(t0) + thrust::get<0>(t1), thrust::get<1>(t0) + thrust::get<1>(t1));
    }
};

// struct get_label
// {
//     int * col;
//     thrust::tuple<int,int> * mem;
//     int * memnnz;
//     int n;
//     int T;
//     //int * _col, int * _mem, int * _memnnz,
//     get_label(int _n, thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int,int> > &_mem, thrust::universal_vector<int> &_memnnz, int _T) {
//         col= thrust::raw_pointer_cast(_col.data());
//         mem = thrust::raw_pointer_cast(_mem.data());
//         memnnz= thrust::raw_pointer_cast(_memnnz.data());
//         //labellist = thrust::raw_pointer_cast(&_labellist[0]);
//         n = _n;
//         T = _T;
//     }
//   __device__
//   thrust::tuple<int,int> operator()(unsigned int thread_id)
//   {
//     int max = 0;
//     int offset = -1;
//     for(int i = 0; i < memnnz[col[i]]; i++){
//       if(thrust::get<1>(mem[col[thread_id] * T + i]) > max) {
//         offset = i;
//         max = thrust::get<1>(mem[col[thread_id] * T + i]);
//       }
//     }
//     thrust::tuple<int, int> val(thrust::get<0>(mem[col[thread_id] * T + offset]), 1);
//     return val;
//   }
// };

struct get_label
{
    int * col;
    thrust::tuple<int,int> * max_per_row;
    //int * _col, int * _mem, int * _memnnz,
    get_label(thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int,int> > &_max_per_row) {
        col= thrust::raw_pointer_cast(_col.data());
        max_per_row = thrust::raw_pointer_cast(_max_per_row.data());
    }
  __device__
  thrust::tuple<int,int> operator()(unsigned int thread_id)
  {
    thrust::tuple<int,int> val(thrust::get<0>(max_per_row[col[thread_id]]), 1);
    return val;
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
    get_frequency(int _n, thrust::universal_vector<int> &_row, thrust::universal_vector<thrust::tuple<int,int> > &_mem, thrust::universal_vector<thrust::tuple<int,int> > &_labellist, thrust::universal_vector<int> &_row_id) {
        row= thrust::raw_pointer_cast(_row.data());
        mem = thrust::raw_pointer_cast(_mem.data());
        labellist = thrust::raw_pointer_cast(_labellist.data());
        row_id = thrust::raw_pointer_cast(_row_id.data());
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
    thrust::tuple<int,int> val(thrust::get<0>(labellist[thread_id]), count);
    return val;
  }
};

struct sort_per_row
{
    int * row_id;
    int * row;
    int * col;
    thrust::tuple<int,int> * labellist;
    int n;
    //int * _col, int * _mem, int * _memnnz,
    sort_per_row(int _n,thrust::universal_vector<int> &_row, thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int,int> > &_labellist, thrust::universal_vector<int> &_row_id) {
        row= thrust::raw_pointer_cast(_row.data());
        col= thrust::raw_pointer_cast(_col.data());
        labellist = thrust::raw_pointer_cast(_labellist.data());
        row_id = thrust::raw_pointer_cast(_row_id.data());
        n = _n;
    }
  __device__
  void operator()(unsigned int thread_id)
  {
    if(thrust::get<1>(labellist[row[row_id[thread_id]]]) < thrust::get<1>(labellist[thread_id])) {
      atomicCAS(&(thrust::get<0>(labellist[row[row_id[thread_id]]])), thrust::get<0>(labellist[row[row_id[thread_id]]]), thrust::get<0>(labellist[thread_id]));
      atomicCAS(&(thrust::get<1>(labellist[row[row_id[thread_id]]])), thrust::get<1>(labellist[row[row_id[thread_id]]]), thrust::get<1>(labellist[thread_id]));
    }
  }
};


struct check_ties
{
    int * row_id;
    int * labellistnnz;
    int * row;
    int * col;
    thrust::tuple<int,int> * labellist;
    int n;
    //int * _col, int * _mem, int * _memnnz,
    check_ties(int _n,thrust::universal_vector<int> &_row, thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int,int> > &_labellist, thrust::universal_vector<int> &_row_id,
    thrust::universal_vector<int> &_labellistnnz) {
        row= thrust::raw_pointer_cast(_row.data());
        col= thrust::raw_pointer_cast(_col.data());
        labellist = thrust::raw_pointer_cast(_labellist.data());
        row_id = thrust::raw_pointer_cast(_row_id.data());
        labellistnnz = thrust::raw_pointer_cast(_labellistnnz.data());
        n = _n;
    }
  __device__
  void operator()(unsigned int thread_id)
  {
    if(thrust::get<1>(labellist[row[row_id[thread_id]]]) == thrust::get<1>(labellist[thread_id]) && thrust::get<0>(labellist[row[row_id[thread_id]]]) != thrust::get<0>(labellist[thread_id])) {
      atomicCAS(&(thrust::get<0>(labellist[row[row_id[thread_id]]+labellistnnz[row_id[thread_id]]])), thrust::get<0>(labellist[row[row_id[thread_id]]+labellistnnz[row_id[thread_id]]]), thrust::get<0>(labellist[thread_id]));
      atomicCAS(&(thrust::get<1>(labellist[row[row_id[thread_id]]+labellistnnz[row_id[thread_id]]])), thrust::get<1>(labellist[row[row_id[thread_id]]+labellistnnz[row_id[thread_id]]]), thrust::get<1>(labellist[thread_id]));
      atomicAdd(&(labellistnnz[row_id[thread_id]]), 1);
    }
  }
};

struct break_ties
{
    int * row_id;
    int * labellistnnz;
    int * row;
    int * col;
    thrust::tuple<int,int> * labellist;
    int n;
    unsigned int seed;
    break_ties(int _n,thrust::universal_vector<int> &_row, thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int,int> > &_labellist, thrust::universal_vector<int> &_row_id,
    thrust::universal_vector<int> &_labellistnnz, unsigned int &_seed) {
        row= thrust::raw_pointer_cast(&_row[0]);
        col= thrust::raw_pointer_cast(&_col[0]);
        labellist = thrust::raw_pointer_cast(&_labellist[0]);
        row_id = thrust::raw_pointer_cast(&_row_id[0]);
        labellistnnz = thrust::raw_pointer_cast(&_labellistnnz[0]);
        n = _n;
        seed = _seed;
    }

  __device__
  void operator()(unsigned int thread_id)
  {
    unsigned int seed2 = hash(seed);
    thrust::default_random_engine rng(seed2);
    //if(thread_id == 12)
    //printf("labellilstnnz %d\n", labellistnnz[thread_id]);
    thrust::uniform_real_distribution<float> u01(0,1);
    float rand2 = u01(rng) * labellistnnz[thread_id];
    int offset = (int)truncf(rand2);
    //if(thread_id == 12)
    //rng.discard(thread_id);
    labellist[row[thread_id]] = labellist[row[thread_id]+offset];
  }
};


// struct update_memory
// {
//     int * memnnz;
//     int * row;
//     thrust::tuple<int,int> * mem;
//     thrust::tuple<int,int> * labellist;
//     int n;
//     int T;
//     //int * _col, int * _mem, int * _memnnz,
//     update_memory(int _n, thrust::universal_vector<int> &_row, thrust::universal_vector<thrust::tuple<int,int> > &_mem, thrust::universal_vector<thrust::tuple<int,int> > &_labellist, thrust::universal_vector<int> &_memnnz, int _T) {
//         row= thrust::raw_pointer_cast(_row.data());
//         mem = thrust::raw_pointer_cast(_mem.data());
//         labellist = thrust::raw_pointer_cast(_labellist.data());
//         memnnz = thrust::raw_pointer_cast(_memnnz.data());
//         n = _n;
//         T = _T;
//     }
//   __device__
//   void operator()(unsigned int thread_id)
//   {
//     int flag = 0;
//     for(int i = 0; i < memnnz[thread_id]; i++) {
//       if(thrust::get<0>(mem[thread_id*T+i]) == thrust::get<0>(labellist[row[thread_id]])){
//         int count = thrust::get<1>(mem[thread_id*T+i])+1;
//         thrust::tuple<int,int> val(thrust::get<0>(mem[thread_id*T+i]), count);
//         mem[thread_id*T+i] = val;
//         flag = 1;
//       }
//     }
//     if(flag == 0 && memnnz[thread_id] < T) {
//       thrust::tuple<int,int> val(thrust::get<0>(labellist[row[thread_id]]), 1);
//       mem[thread_id*T+memnnz[thread_id]] = val;
//       memnnz[thread_id]++;
//     }
//     else if(flag == 0 && memnnz[thread_id] == T) {
//       int lowcount = 1000000000;
//       int index = -1;
//       for(int i = 0; i < n; i++) {
//         if(thrust::get<1>(mem[thread_id*T+i]) < lowcount) {
//           index = i;
//           lowcount = thrust::get<1>(mem[thread_id*T+i]);
//         }
//       }
//       thrust::tuple<int,int> val(thrust::get<0>(labellist[row[thread_id]]), 1);
//       mem[thread_id*T+index] = val;
//     }
//   }
// };

struct update_memory
{
    int * memnnz;
    int * row;
    thrust::tuple<int,int> * mem;
    thrust::tuple<int,int> * labellist;
    int n;
    int T;
    thrust::tuple<int, int> * max_per_row;
    //int * _col, int * _mem, int * _memnnz,
    update_memory(int _n, thrust::universal_vector<int> &_row, thrust::universal_vector<thrust::tuple<int,int> > &_mem, thrust::universal_vector<thrust::tuple<int,int> > &_labellist, thrust::universal_vector<int> &_memnnz, int _T,
    thrust::universal_vector<thrust::tuple<int,int>> &_max_per_row) {
        row= thrust::raw_pointer_cast(_row.data());
        mem = thrust::raw_pointer_cast(_mem.data());
        labellist = thrust::raw_pointer_cast(_labellist.data());
        memnnz = thrust::raw_pointer_cast(_memnnz.data());
        n = _n;
        T = _T;
        max_per_row = thrust::raw_pointer_cast(_max_per_row.data());
    }
  __device__
  void operator()(unsigned int thread_id)
  {
    int flag = 0;
    for(int i = 0; i < memnnz[thread_id]; i++) {
      if(thrust::get<0>(mem[thread_id*T+i]) == thrust::get<0>(labellist[row[thread_id]])){
        int count = thrust::get<1>(mem[thread_id*T+i])+1;
        thrust::tuple<int,int> val(thrust::get<0>(mem[thread_id*T+i]), count);
        mem[thread_id*T+i] = val;
        if(thrust::get<0>(mem[thread_id*T+i]) != thrust::get<0>(max_per_row[thread_id]) && thrust::get<1>(mem[thread_id*T+i]) > thrust::get<1>(max_per_row[thread_id])) {
            max_per_row[thread_id] = mem[thread_id*T+i];
        }
        flag = 1;
      }
    }
    if(flag == 0 && memnnz[thread_id] < T) {
      thrust::tuple<int,int> val(thrust::get<0>(labellist[row[thread_id]]), 1);
      mem[thread_id*T+memnnz[thread_id]] = val;
      // if(thrust::get<0>(mem[thread_id*T+memnnz[thread_id]]) != thrsut::get<0>(max_per_row[thread_id]) && thrust::get<1>(mem[thread_id*T+memnnz[thread_id]]) > thrsut::get<1>(max_per_row[thread_id])) {
      //       max_per_row[thread_id] = mem[thread_id*T+memnnz[thread_id]];
      // }
      memnnz[thread_id]++;
    }
    else if(flag == 0 && memnnz[thread_id] == T) {
      int lowcount = 1000000000;
      int index = -1;
      for(int i = 0; i < n; i++) {
        if(thrust::get<1>(mem[thread_id*T+i]) < lowcount) {
          index = i;
          lowcount = thrust::get<1>(mem[thread_id*T+i]);
        }
      }
      thrust::tuple<int,int> val(thrust::get<0>(labellist[row[thread_id]]), 1);
      mem[thread_id*T+index] = val;
    }
  }
};


struct post_process
{
    thrust::tuple<int,int> * mem;
    int T;
    int n;
    float r;
    //int * _col, int * _mem, int * _memnnz,
    post_process(int _n, thrust::universal_vector<thrust::tuple<int,int> > &_mem, int _T, float _r) {
        mem = thrust::raw_pointer_cast(_mem.data());
        n = _n;
        T = _T;
        r = _r;
    }
  __device__
  thrust::tuple<int,int> operator()(unsigned int thread_id)
  {
    thrust::tuple<int, int> res(-1,0);
    if(thrust::get<1>(mem[thread_id]) < (float)(T*r)) {
      return res;
    }
    return mem[thread_id];
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
   int T = 50;
    float r = 0.1;
    int n = (int)graph_view.number_of_vertices(); 
    int cols = (int)graph_view.local_edge_partition_view().number_of_edges();
    int mod = 1;
    int num_lines = 0;

    vertex_t * rows = graph_view.local_edge_partition_view().indices();
    edge_t * columns = graph_view.local_edge_partition_view().offsets();

    thrust::universal_vector<int> row(n+1);
    thrust::universal_vector<int> col(cols);

    for(int i = 0; i < n+1; i++) {
      row[i] = (int)rows[i];
    }

    for(int j = 0; j < cols; j++) {
      cols[j] = (int)columns[j];
    }
    std::cout << "finished creating\n" << std::endl;

    thrust::universal_vector<int> row_id(cols);
     for(int i = 0; i < n; i++) {
        for(int j = row[i]; j < row[i+1]; j++) {
            row_id[j] = i;
        }
    }
    std::cout << "finished creating row_id\n" << std::endl;

    int ne = T;
    thrust::universal_vector<thrust::tuple<int,int> > mem(n*ne);
    //std::cout << mem.size() << std::endl;
    thrust::universal_vector<int> memnnz(n);
    for(int i = 0; i < n; i++) {
            thrust::tuple<int,int> init(i, 1);
        for(int j = 0; j < ne; j++) {
                mem[i*ne+j] = init;
                //std::cout << "this is i " << i << "this is j " << j << " " << thrust::get<0>(mem[i*ne+j]) << " " << thrust::get<1>(mem[i*ne+j]) << std::endl;
        }
    }
    std::cout << "finished creating mem\n" << std::endl;
    thrust::fill(memnnz.begin(), memnnz.end(), 1);
    thrust::universal_vector<thrust::tuple<int,int> > labellist(cols);
    thrust::fill(labellist.begin(), labellist.end(), 0);

    std::cout << "finished creating memnnz labellsit\n" << std::endl;
    double times = 0;
    double timel1 = 0;
    double timel2 = 0;
    double timel2_1 = 0;
    double timel2_2 = 0;
    double timel3 = 0;
    double timepp = 0;
    unsigned int seed;
    thrust::universal_vector<thrust::tuple<int,int>> max_per_row(n);
    for(int i = 0; i < n; i++) {
      max_per_row[i] = thrust::make_tuple(i, 1);
    }
     auto start = std::chrono::high_resolution_clock::now();
    for(int k = 1; k < T; k++) {
        seed = k;
        auto starts = std::chrono::high_resolution_clock::now();
        get_label s(col, max_per_row);
        thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols), labellist.begin(), s);
        cudaDeviceSynchronize();
        auto ends = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffs = ends- starts;
        times += diffs.count();
        //std::cout << "Time do speaker is "
        //          <<  diffs.count() << " s\n";
        // if(k%mod == 0) {
        //     std::cout << "Speaker labellist \n";
        //     for(int i = row[12]; i < row[13]; i++) {
        //         std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
        //     }
        //     std::cout << "\n";
        // }
        auto startl1 = std::chrono::high_resolution_clock::now();
        get_frequency l1(n, row, mem, labellist, row_id);
        thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols), labellist.begin(), l1);
        cudaDeviceSynchronize();
        auto endl1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffl1 = endl1 - startl1;
        timel1 += diffl1.count();
        //std::cout << "Time do listener1 is "
        //          <<  diffl1.count() << " s\n";
        // if(k%mod == 0) {
        //     std::cout << "listener 1 labellist \n";
        //     for(int i = row[12]; i < row[13]; i++) {
        //         std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
        //     }
        //     std::cout << "\n";
        // }
        auto startl2 = std::chrono::high_resolution_clock::now();
        sort_per_row l2(n, row, col, labellist, row_id);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols), l2);
        cudaDeviceSynchronize();
        auto endl2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffl2 = endl2 - startl2;
        timel2 += diffl2.count();
        //std::cout << "Time do listener2 is "
        //          <<  diffl2.count() << " s\n";
        // if(k%mod == 0) {
        //    std::cout << "listener 2 labellist \n";
        //     for(int i = row[12]; i < row[13]; i++) {
        //         std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
        //     }
        //     std::cout << "\n";
        // }
        thrust::universal_vector<int> labellistnnz(n);
        thrust::fill(labellistnnz.begin(), labellistnnz.end(), 1);
        auto startl2_1 = std::chrono::high_resolution_clock::now();
        check_ties l2_1(n, row, col, labellist, row_id, labellistnnz);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols), l2_1);
        cudaDeviceSynchronize();
        auto endl2_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffl2_1 = endl2_1 - startl2_1;
        timel2_1 += diffl2_1.count();
        //std::cout << "Time do listener2_1 is "
        //          <<  diffl2_1.count() << " s\n";
        // if(k%mod == 0) {
        //    std::cout << "listener 2_1 labellist \n";
        //     for(int i = row[12]; i < row[13]; i++) {
        //         std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
        //     }
        //     std::cout << "\n";
        // }

        auto startl2_2 = std::chrono::high_resolution_clock::now();
        break_ties l2_2(n, row, col, labellist, row_id, labellistnnz, seed);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), l2_2);
        cudaDeviceSynchronize();
        auto endl2_2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffl2_2 = endl2_2 - startl2_2;
        timel2_2 += diffl2_2.count();
        //std::cout << "Time do listener2_2 is "
        //          <<  diffl2_2.count() << " s\n";
        // if(k%mod == 0) {
        //     std::cout << "listener 2_2 labellist \n";
        //     for(int i = row[12]; i < row[13]; i++) {
        //         std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
        //     }
        //     std::cout << "\n";
        // }
        auto startl3 = std::chrono::high_resolution_clock::now();
        update_memory l3(n, row, mem, labellist, memnnz, ne, max_per_row);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), l3);
        cudaDeviceSynchronize();
        auto endl3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffl3 = endl3 - startl3;
        timel3 += diffl3.count();
        //std::cout << "Time do listener3 is "
        //          <<  diffl3.count() << " s\n";
        // if(k%mod == 0) {
        //     std::cout << "Listener 3 mem\n";
        //     for(int i = 0; i < n; i++) {
        //         std::cout << "Vertex "<< i << ": ";
        //         for(int j = 0; j < memnnz[i]; j++) {
        //             std::cout << thrust::get<0>(mem[i*T+j]) << " " << thrust::get<1>(mem[i*T+j]) << " ";
        //         }
        //         std::cout << "\n";
        //     }
        //     std::cout << "\n";
        // }
    }
    auto startpp = std::chrono::high_resolution_clock::now();
    post_process p(n, mem, T, r);
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n*ne), mem.begin(), p);
    cudaDeviceSynchronize();
        auto endpp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffpp = endpp - startpp;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "\nTime do total is "
                  <<  diff.count() << " s\n";

        std::cout << "average time for speaker is "
                  <<  times/(double)T << " s\n";
        std::cout << "average time for listerner 1 is "
                  <<  timel1/(double)T << " s\n";
        std::cout << "average time for listerner 2 is "
                  <<  timel2/(double)T << " s\n";
        std::cout << "average time for listerner 2_1 is "
                  <<  timel2_1/(double)T << " s\n";
        std::cout << "average time for listerner 2_2 is "
                  <<  timel2_2/(double)T << " s\n";
        std::cout << "average time for listerner 3 is "
                  <<  timel3/(double)T << " s\n";
        std::cout << "time for post is "
                  <<  diffpp.count() << " s\n";
        /*for(int i = 0; i < n; i++) {
        int count = 0;
        std::cout << "Vertex " << i << " is in communities ";
        for(int j = 0; j < memnnz[i]; j++) {
            if(thrust::get<0>(mem[i*T+j]) != -1) {
                std::cout << thrust::get<0>(mem[i*T+j]) << " ";
                count++;
            }
        }
        if(count > 1)
            std::cout << "overlap\n";
        else
            std::cout << " only community\n";
    }*/
    return;
}
}
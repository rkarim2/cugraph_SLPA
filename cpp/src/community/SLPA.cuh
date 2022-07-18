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
    get_label(int _n, thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int,int> > &_mem, thrust::universal_vector<int> &_memnnz) {
        col= thrust::raw_pointer_cast(_col.data());
        mem = thrust::raw_pointer_cast(_mem.data());
        memnnz= thrust::raw_pointer_cast(_memnnz.data());
        //labellist = thrust::raw_pointer_cast(&_labellist[0]);
        n = _n;

    }
  __device__
  thrust::tuple<int,int> operator()(unsigned int thread_id)
  {
    int max = 0;
    int offset = -1;
    for(int i = 0; i < memnnz[col[i]]; i++){
      if(thrust::get<1>(mem[col[thread_id] * n + i]) > max) {
        offset = i;
        max = thrust::get<1>(mem[col[thread_id] * n + i]);
      }
    }
    thrust::tuple<int, int> val(thrust::get<0>(mem[col[thread_id] * n + offset]), 1);
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
    break_ties(int _n,thrust::universal_vector<int> _row, thrust::universal_vector<int> _col, thrust::universal_vector<thrust::tuple<int,int> > _labellist, thrust::universal_vector<int> _row_id,
    thrust::universal_vector<int> _labellistnnz, unsigned int _seed) {
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
    //unsigned int seed = hash(thread_id);
    thrust::default_random_engine rng(seed);
    if(thread_id == 12)
    printf("labellilstnnz %d\n", labellistnnz[thread_id]);
    thrust::uniform_real_distribution<float> u01(0,labellistnnz[thread_id]);
    if(thread_id == 12)
    printf("random call %d\n", (int)u01(rng));
    labellist[row[thread_id]] = labellist[row[thread_id]+(int)u01(rng)];
  }
};


struct update_memory 
{
    int * memnnz;
    int * row;
    thrust::tuple<int,int> * mem;
    thrust::tuple<int,int> * labellist;
    int n;
    //int * _col, int * _mem, int * _memnnz,
    update_memory(int _n, thrust::universal_vector<int> &_row, thrust::universal_vector<thrust::tuple<int,int> > &_mem, thrust::universal_vector<thrust::tuple<int,int> > &_labellist, thrust::universal_vector<int> &_memnnz) {
        row= thrust::raw_pointer_cast(_row.data());
        mem = thrust::raw_pointer_cast(_mem.data());
        labellist = thrust::raw_pointer_cast(_labellist.data());
        memnnz = thrust::raw_pointer_cast(_memnnz.data());
        n = _n;
    }
  __device__
  void operator()(unsigned int thread_id)
  {
    int flag = 0;
    for(int i = 0; i < memnnz[thread_id]; i++) {
      if(thrust::get<0>(mem[thread_id*n+i]) == thrust::get<0>(labellist[row[thread_id]])){
        int count = thrust::get<1>(mem[thread_id*n+i])+1;
        thrust::tuple<int,int> val(thrust::get<0>(mem[thread_id*n+i]), count);
        mem[thread_id*n+i] = val;
        flag = 1;
      }
    }
    if(flag == 0 && memnnz[thread_id] < n) {
      thrust::tuple<int,int> val(thrust::get<0>(labellist[row[thread_id]]), 1);
      mem[thread_id*n+memnnz[thread_id]] = val;
      memnnz[thread_id]++;
    }
    else if(flag == 0 && memnnz[thread_id] == n) {
      int lowcount = 1000000000;
      int index = -1;
      for(int i = 0; i < n; i++) {
        if(thrust::get<1>(mem[thread_id*n+i]) < lowcount) {
          index = i;
          lowcount = thrust::get<1>(mem[thread_id*n+i]);
        }
      }
      thrust::tuple<int,int> val(thrust::get<0>(labellist[row[thread_id]]), 1);
      mem[thread_id*n+index] = val;
    }
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
    int n = 13;
    int cols = 56;
    int mod = 1;
    thrust::universal_vector<int> row(n+1);
    row[0] = 0;
    row[1] = 3;
    row[2] = 7;
    row[3] = 12;
    row[4] = 17;
    row[5] = 21;
    row[6] = 26;
    row[7] = 31;
    row[8] = 34;
    row[9] = 38;
    row[10] = 42;
    row[11] = 46;
    row[12] = 49;
    row[13] = 56;
    thrust::universal_vector<int> col(cols);
    col[0] = 1;
    col[1] = 2;
    col[2] = 3;
    col[3] = 0;
    col[4] = 2;
    col[5] = 3;
    col[6] = 12;
    col[7] = 0;
    col[8] = 1;
    col[9] = 3;
    col[10] = 6;
    col[11] = 12;
    col[12] = 0;
    col[13] = 1;
    col[14] = 2;
    col[15] = 8;
    col[16] = 12;
    col[17] = 5;
    col[18] = 6;
    col[19] = 7;
    col[20] = 12;
    col[21] = 4;
    col[22] = 6;
    col[23] = 7;
    col[24] = 9;
    col[25] = 12;
    col[26] = 2;
    col[27] = 4;
    col[28] = 5;
    col[29] = 7;
    col[30] = 12;
    col[31] = 4;
    col[32] = 5;
    col[33] = 6;
    col[34] = 3;
    col[35] = 9;
    col[36] = 10;
    col[37] = 11;
    col[38] = 5;
    col[39] = 8;
    col[40] = 10;
    col[41] = 11;
    col[42] = 8;
    col[43] = 9;
    col[44] = 11;
    col[45] = 12;
    col[46] = 8;
    col[47] = 9;
    col[48] = 10;
    col[49] = 1;
    col[50] = 2;
    col[51] = 3;
    col[52] = 4;
    col[53] = 5;
    col[54] = 6;
    col[55] = 10;
    thrust::universal_vector<int> row_id(cols);
    row_id[0] = 0;    
    row_id[1] = 0;    
    row_id[2] = 0;    
    row_id[3] = 1;    
    row_id[4] = 1;    
    row_id[5] = 1;    
    row_id[6] = 1;
    row_id[7] = 2;    
    row_id[8] = 2;    
    row_id[9] = 2;    
    row_id[10] = 2;
    row_id[11] = 2;
    row_id[12] = 3;
    row_id[13] = 3;
    row_id[14] = 3;
    row_id[15] = 3;
    row_id[16] = 3;
    row_id[17] = 4;
    row_id[18] = 4;
    row_id[19] = 4;
    row_id[20] = 4;
    row_id[21] = 5;
    row_id[22] = 5;
    row_id[23] = 5;
    row_id[24] = 5;
    row_id[25] = 5;
    row_id[26] = 6;
    row_id[27] = 6;
    row_id[28] = 6;
    row_id[29] = 6;
    row_id[30] = 6;
    row_id[31] = 7;
    row_id[32] = 7;
    row_id[33] = 7;
    row_id[34] = 8;
    row_id[35] = 8;
    row_id[36] = 8;
    row_id[37] = 8;
    row_id[38] = 9;
    row_id[39] = 9;
    row_id[40] = 9;
    row_id[41] = 9;
    row_id[42] = 10;
    row_id[43] = 10;
    row_id[44] = 10;
    row_id[45] = 10;
    row_id[46] = 11;
    row_id[47] = 11;
    row_id[48] = 11;
    row_id[49] = 12;
    row_id[50] = 12;
    row_id[51] = 12;
    row_id[52] = 12;
    row_id[53] = 12;
    row_id[54] = 12;
    row_id[55] = 12;
    thrust::universal_vector<thrust::tuple<int,int> > mem(n*n);
    thrust::universal_vector<int> memnnz(n);
    for(int i = 0; i < n; i++) {
        thrust::tuple<int,int> init(i, 1);
        thrust::fill(mem.begin()+i*n, mem.begin()+(i*n+n), init);
    }
    thrust::fill(memnnz.begin(), memnnz.end(), 1);
    thrust::universal_vector<thrust::tuple<int,int> > labellist(cols);
    thrust::fill(labellist.begin(), labellist.end(), 0);

    unsigned int seed;
    for(int k = 0; k < T; k++) {
        seed = hash(k);
        get_label s(n, col, mem, memnnz);
        thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols), labellist.begin(), s);
        cudaDeviceSynchronize();
        if(k%mod == 0) {
            std::cout << "Speaker labellist \n";
            for(int i = row[12]; i < row[13]; i++) {
                std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
            }
            std::cout << "\n";
        }
        get_frequency l1(n, row, mem, labellist, row_id);
        thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols), labellist.begin(), l1);
        cudaDeviceSynchronize();
        if(k%mod == 0) {
            std::cout << "listener 1 labellist \n";
            for(int i = row[12]; i < row[13]; i++) {
                std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
            }
            std::cout << "\n";
        }
        sort_per_row l2(n, row, col, labellist, row_id);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols), l2);
        cudaDeviceSynchronize();
        if(k%mod == 0) {
           std::cout << "listener 2 labellist \n";
            for(int i = row[12]; i < row[13]; i++) {
                std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
            }
            std::cout << "\n";
        }
        thrust::universal_vector<int> labellistnnz(n);
        thrust::fill(labellistnnz.begin(), labellistnnz.end(), 1);
        check_ties l2_1(n, row, col, labellist, row_id, labellistnnz);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols), l2_1);
        cudaDeviceSynchronize();
        if(k%mod == 0) {
           std::cout << "listener 2_1 labellist \n";
            for(int i = row[12]; i < row[13]; i++) {
                std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
            }
            std::cout << "\n";
        }
        
        break_ties l2_2(n, row, col, labellist, row_id, labellistnnz, seed);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), l2_2);
        cudaDeviceSynchronize();
        if(k%mod == 0) {
            std::cout << "listener 2_2 labellist \n";
            for(int i = row[12]; i < row[13]; i++) {
                std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
            }
            std::cout << "\n";
        }
        update_memory l3(n, row, mem, labellist, memnnz);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), l3);
        cudaDeviceSynchronize();
        if(k%mod == 0) {
            std::cout << "Listener 3 mem\n";
            for(int i = 0; i < n; i++) {
                std::cout << "Vertex "<< i << ": ";
                for(int j = 0; j < memnnz[i]; j++) {
                    std::cout << thrust::get<0>(mem[i*n+j]) << " " << thrust::get<1>(mem[i*n+j]) << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    post_process p(n, mem, T, r);
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n*n), mem.begin(), p);
    cudaDeviceSynchronize();
     for(int i = 0; i < n; i++) {
        int count = 0;
        std::cout << "Vertex " << i << " is in communities ";
        for(int j = 0; j < memnnz[i]; j++) {
            if(thrust::get<0>(mem[i*n+j]) != -1) {
                std::cout << thrust::get<0>(mem[i*n+j]) << " ";
                count++;
            }
        }
        if(count > 1)
            std::cout << "overlap\n";
        else
            std::cout << " only community\n";
    }
    return;
}
}
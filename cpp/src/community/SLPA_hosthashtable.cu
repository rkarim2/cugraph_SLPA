#include <cuda.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/universal_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <chrono>
#include <cuco/static_reduction_map.cuh>
#include <cuco/static_map.cuh>


struct custom_key_type {
  int32_t a;
  int32_t b;

  __host__ __device__ custom_key_type() {}
  __host__ __device__ custom_key_type(int32_t x) : a{x}, b{x} {}
  __host__ __device__ custom_key_type(int32_t x, int32_t y) : a{x}, b{y} {}
  __device__ bool operator==(custom_key_type const& other) const
  {
    return a == other.a and b == other.b;
  }
};

struct custom_hash {
  template <typename key_type>
  __device__ uint32_t operator()(key_type k)
  {
    return k.a;
  };
};

struct custom_key_equals {
  template <typename key_type>
  __device__ bool operator()(key_type const& lhs, key_type const& rhs)
  {
    return lhs.a == rhs.a;
  }
};


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

template <typename MapType, std::size_t Capacity>
__global__ void send_aggregate_label(int n, int * row, int * col, int * row_id, thrust::tuple<uint32_t,int> * max_per_row, int * key_found_nnz, thrust::tuple<uint32_t,int> * key_found)
{
  using Key   = typename MapType::key_type;
  using Value = typename MapType::mapped_type;

  namespace cg = cooperative_groups;
  // define a mutable view for insert operations
  using mutable_view_type = typename MapType::device_mutable_view<>;
  // define a immutable view for find/contains operations
  using view_type = typename MapType::device_view<>;

// hash table storage in shared memory
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ typename mutable_view_type::slot_type slots[Capacity];

  // construct the table from the provided array in shared memory
  auto map = mutable_view_type::make_from_uninitialized_slots(cg::this_thread_block(), &slots[0], Capacity, -1);

  auto g            = cg::this_thread_block();
  std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  int rank          = g.thread_rank();
  
  //printf("hello from thread %d\n", rank);
  // insert {thread_rank, thread_rank} for each thread in thread-block
  //for(int i = index; i < Capacity; i+=256)
  if(index < row[n]) {
  uint32_t hi = (uint32_t)row_id[index];
  uint32_t low = thrust::get<0>(max_per_row[col[index]]);
  int64_t encode = (int64_t) hi << 32 | low;
  //printf("hello from thread %d\n", index);
  map.insert(cuco::pair<Key, Value>(encode, 1));
  }
  g.sync();

  auto find_map = view_type(map);
  auto us = find_map.get_slots();
  for(int i = rank; i < Capacity; i+= blockDim.x) {
    if(us[i].second >= 1) {
      uint32_t key = (uint32_t) (us[i].first & 0xffffffff);
      uint32_t range = (uint32_t) (us[i].first >> 32);
      int res = atomicAdd(&(key_found_nnz[range]), 1);
      key_found[row[range]+res] = thrust::make_tuple<uint32_t, int>(key, us[i].second);
    }
  }
}


struct check_duplicates
{
    int * row_id;
    int * row;
    thrust::tuple<uint32_t,int> * labellist;
    int n;
    //int * _col, int * _mem, int * _memnnz,
    check_duplicates(int _n, thrust::universal_vector<int> &_row, thrust::universal_vector<thrust::tuple<uint32_t,int> > &_labellist, thrust::universal_vector<int> &_row_id) {
        row= thrust::raw_pointer_cast(_row.data());
        labellist = thrust::raw_pointer_cast(_labellist.data());
        row_id = thrust::raw_pointer_cast(_row_id.data());
        n = _n;
    }
  __device__
  thrust::tuple<int,int> operator()(unsigned int thread_id)
  {

    // int offset2 = row[row_id[thread_id]+1] - labellistnnz[offset1];
  if(thrust::get<1>(labellist[thread_id]) != 0) {
    int offset1 = row[row_id[thread_id]];
    int offset2 = row[row_id[thread_id]+1];
    int sub = thrust::get<1>(labellist[thread_id]);
    for(int i = offset1; i < offset2; i++) {
      if(thrust::get<0>(labellist[thread_id]) == thrust::get<0>(labellist[i])) {
        thrust::get<1>(labellist[thread_id]) += thrust::get<1>(labellist[i]);
      }
    }
    thrust::get<1>(labellist[thread_id]) -= sub;
    }
    // thrust::tuple<int,int> val(thrust::get<0>(labellist[thread_id]), count);
    // return val;
  }
};


// struct get_label
// {
//     int * col;
//     thrust::tuple<int32_t,int> * max_per_row;
//     int * row_id;
//     //int * _col, int * _mem, int * _memnnz,
//     get_label(thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int32_t,int> > &_max_per_row, thrust::universal_vector<int> &_row_id) {
//         col= thrust::raw_pointer_cast(_col.data());
//         max_per_row = thrust::raw_pointer_cast(_max_per_row.data());
//         row_id = thrust::raw_pointer_cast(_row_id.data());
//     }
//   __device__
//   cuco::pair<custom_key_type,int> operator()(unsigned int thread_id)
//   {
//     // thrust::tuple<custom_key_type,int> val(custom_key_type{row_id[thread_id], thrust::get<0>(max_per_row[col[thread_id]])}, 1);
//     // return val;
//     return cuco::make_pair<custom_key_type, int>(custom_key_type{row_id[thread_id], thrust::get<0>(max_per_row[col[thread_id]])}, 1);
//   }
// };

struct write_back
{
    int * row;
    int * labellistnnz;
    thrust::tuple<int32_t, int> * labellist;
    cuco::pair<cuda::__4::atomic<custom_key_type, cuda::std::__4::__detail::thread_scope_device>, cuda::__4::atomic<int, cuda::std::__4::__detail::thread_scope_device>> * slots;
    //int * _col, int * _mem, int * _memnnz,
    write_back(thrust::universal_vector<int> &_row, thrust::universal_vector<int> &_labellistnnz, thrust::universal_vector<thrust::tuple<int32_t, int>> &_labellist, cuco::static_reduction_map<cuco::reduce_add<int>, custom_key_type, int> &_map) {
        row= thrust::raw_pointer_cast(_row.data());
        labellistnnz = thrust::raw_pointer_cast(_labellistnnz.data());
        labellist = thrust::raw_pointer_cast(_labellist.data());
        slots = _map.get_device_view().get_slots();
    }
  __device__
  void operator()(unsigned int thread_id)
  {
     if(slots[thread_id].second >= 1) {
      int32_t key = slots[thread_id].first.load().b;

      int32_t range = slots[thread_id].first.load().a;
      int res = atomicAdd(&(labellistnnz[range]), 1);
      labellist[row[range]+res] = thrust::make_tuple<int32_t, int>(key, slots[thread_id].second.load());
    }
  }
};


struct get_frequency
{
    int * row_id;
    int * row;
    thrust::tuple<uint32_t,int> * mem;
    thrust::tuple<uint32_t,int> * labellist;
    int n;
    //int * _col, int * _mem, int * _memnnz,
    get_frequency(int _n, thrust::universal_vector<int> &_row, thrust::universal_vector<thrust::tuple<uint32_t,int> > &_mem, thrust::universal_vector<thrust::tuple<uint32_t,int> > &_labellist, thrust::universal_vector<int> &_row_id) {
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
    thrust::tuple<int32_t,int> * labellist;
    int n;
    //int * _col, int * _mem, int * _memnnz,
    sort_per_row(int _n,thrust::universal_vector<int> &_row, thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int32_t,int> > &_labellist, thrust::universal_vector<int> &_row_id) {
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
    thrust::tuple<int32_t,int> * labellist;
    int n;
    //int * _col, int * _mem, int * _memnnz,
    check_ties(int _n,thrust::universal_vector<int> &_row, thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int32_t,int> > &_labellist, thrust::universal_vector<int> &_row_id,
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
    thrust::tuple<int32_t,int> * labellist;
    int n;
    unsigned int seed;
    break_ties(int _n,thrust::universal_vector<int> &_row, thrust::universal_vector<int> &_col, thrust::universal_vector<thrust::tuple<int32_t,int> > &_labellist, thrust::universal_vector<int> &_row_id,
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
    //printf("labellistnnz %d\n", labellistnnz[thread_id]);
    thrust::uniform_real_distribution<float> u01(0,1);
    float rand2 = u01(rng) * labellistnnz[thread_id];
    int offset = (int)truncf(rand2);
    //if(thread_id == 12)
    //rng.discard(thread_id);
    labellist[row[thread_id]] = labellist[row[thread_id]+offset];
  }
};


struct update_memory
{
    int * memnnz;
    int * row;
    thrust::tuple<int32_t,int> * mem;
    thrust::tuple<int32_t,int> * labellist;
    int n;
    int T;
    thrust::tuple<int32_t, int> * max_per_row;
    //int * _col, int * _mem, int * _memnnz,
    update_memory(int _n, thrust::universal_vector<int> &_row, thrust::universal_vector<thrust::tuple<int32_t,int> > &_mem, thrust::universal_vector<thrust::tuple<int32_t,int> > &_labellist, thrust::universal_vector<int> &_memnnz, int _T,
    thrust::universal_vector<thrust::tuple<int32_t,int>> &_max_per_row) {
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
    thrust::tuple<int32_t,int> * mem;
    int T;
    int n;
    float r;
    //int * _col, int * _mem, int * _memnnz,
    post_process(int _n, thrust::universal_vector<thrust::tuple<int32_t,int> > &_mem, int _T, float _r) {
        mem = thrust::raw_pointer_cast(_mem.data());
        n = _n;
        T = _T;
        r = _r;
    }
  __device__
  thrust::tuple<int32_t,int> operator()(unsigned int thread_id)
  {
    thrust::tuple<int32_t, int> res(-1,0);
    if(thrust::get<1>(mem[thread_id]) < (float)(T*r)) {
      return res;
    }
    return mem[thread_id];
  }
};

void convertToCSR(char * argv[], thrust::universal_vector<int> * row2, thrust::universal_vector<int> * col2, int &n, int & cols) {
    std::string arg1(argv[1]);
    std::ifstream file(arg1);
    while (file.peek() == '%') file.ignore(2048, '\n');
    int num_lines = 0;
    // Read number of rows and columns
    file >> n >> cols >> num_lines;
    std::cout << "# rows is " << n << " # cols is " << cols << " # lines is " << num_lines << "\n";

    thrust::host_vector<int> * r = new thrust::host_vector<int>();
    thrust::host_vector<int> * c = new thrust::host_vector<int>();
    for (int l = 0; l < num_lines; l++) {
        int rowv, colv;
        file >> rowv >> colv;
        r->push_back(rowv-1);
        c->push_back(colv-1);
        r->push_back(colv-1);
        c->push_back(rowv-1);
    }

    std::cout << r->size() << std::endl;
    file.close();
    //std::cout << r->at(0) << " " <<  r->at(1) << "\n";
    thrust::sort_by_key(thrust::host, (*r).begin(), (*r).end(), (*c).begin());
    std::cout << "finshied sort" << std::endl;

    cols = num_lines*2;

    thrust::host_vector<int> row(n+1);
    row[0] = 0;
    thrust::host_vector<int> col(cols);
    int x = 0;
    int count = 0;
    for(int i = 0; i < n; i++) {
        while((*r)[x] == i) {
            count++;
            x++;
        }
        row[i+1] = count + row[i];
        count = 0;
    }
    for(int i = 0; i < c->size(); i++) {
        col[i] = (*c)[i];
    }
    *row2 = row;
    *col2 = col;
}

int main(int argc, char *argv[]) {
  using Key   = int64_t;
  using Value = int;
  int N = 1024;
  // define the capacity of the map
  static constexpr int capacity = 2048;
   using map_type =
    cuco::static_reduction_map<cuco::reduce_add<Value>, Key, Value, cuda::thread_scope_block>;

    auto const empty_key_sentinel = custom_key_type{-1};
    auto const empty_value_sentinel = -1;
    int T = 50;
    float r = 0.1;
    int n = 13;
    int cols = 56;
    int mod = 1;
    int num_lines = 0;

    // cuco::static_map<custom_key_type, Value> map{
    // cols, 
    // empty_key_sentinel,
    // empty_value_sentinel};

    thrust::universal_vector<int> row;
    // row[0] = 0;
    // row[1] = 3;
    // row[2] = 7;
    // row[3] = 12;
    // row[4] = 17;
    // row[5] = 21;
    // row[6] = 26;
    // row[7] = 31;
    // row[8] = 34;
    // row[9] = 38;
    // row[10] = 42;
    // row[11] = 46;
    // row[12] = 49;
    // row[13] = 56;
    thrust::universal_vector<int> col;
    thrust::universal_vector<double> val(cols);
    // col[0] = 1;
    // col[1] = 2;
    // col[2] = 3;
    // col[3] = 0;
    // col[4] = 2;
    // col[5] = 3;
    // col[6] = 12;
    // col[7] = 0;
    // col[8] = 1;
    // col[9] = 3;
    // col[10] = 6;
    // col[11] = 12;
    // col[12] = 0;
    // col[13] = 1;
    // col[14] = 2;
    // col[15] = 8;
    // col[16] = 12;
    // col[17] = 5;
    // col[18] = 6;
    // col[19] = 7;
    // col[20] = 12;
    // col[21] = 4;
    // col[22] = 6;
    // col[23] = 7;
    // col[24] = 9;
    // col[25] = 12;
    // col[26] = 2;
    // col[27] = 4;
    // col[28] = 5;
    // col[29] = 7;
    // col[30] = 12;
    // col[31] = 4;
    // col[32] = 5;
    // col[33] = 6;
    // col[34] = 3;
    // col[35] = 9;
    // col[36] = 10;
    // col[37] = 11;
    // col[38] = 5;
    // col[39] = 8;
    // col[40] = 10;
    // col[41] = 11;
    // col[42] = 8;
    // col[43] = 9;
    // col[44] = 11;
    // col[45] = 12;
    // col[46] = 8;
    // col[47] = 9;
    // col[48] = 10;
    // col[49] = 1;
    // col[50] = 2;
    // col[51] = 3;
    // col[52] = 4;
    // col[53] = 5;
    // col[54] = 6;
    // col[55] = 10;


    convertToCSR(argv, &row, &col, n, cols);

    std::cout << "finished creating\n" << std::endl;

    thrust::universal_vector<int> row_id(cols);
     for(int i = 0; i < n; i++) {
        for(int j = row[i]; j < row[i+1]; j++) {
            row_id[j] = i;
        }
    }
    std::cout << "finished creating row_id\n" << std::endl;
    // row_id[0] = 0;
    // row_id[1] = 0;
    // row_id[2] = 0;
    // row_id[3] = 1;
    // row_id[4] = 1;
    // row_id[5] = 1;
    // row_id[6] = 1;
    // row_id[7] = 2;
    // row_id[8] = 2;
    // row_id[9] = 2;
    // row_id[10] = 2;
    // row_id[11] = 2;
    // row_id[12] = 3;
    // row_id[13] = 3;
    // row_id[14] = 3;
    // row_id[15] = 3;
    // row_id[16] = 3;
    // row_id[17] = 4;
    // row_id[18] = 4;
    // row_id[19] = 4;
    // row_id[20] = 4;
    // row_id[21] = 5;
    // row_id[22] = 5;
    // row_id[23] = 5;
    // row_id[24] = 5;
    // row_id[25] = 5;
    // row_id[26] = 6;
    // row_id[27] = 6;
    // row_id[28] = 6;
    // row_id[29] = 6;
    // row_id[30] = 6;
    // row_id[31] = 7;
    // row_id[32] = 7;
    // row_id[33] = 7;
    // row_id[34] = 8;
    // row_id[35] = 8;
    // row_id[36] = 8;
    // row_id[37] = 8;
    // row_id[38] = 9;
    // row_id[39] = 9;
    // row_id[40] = 9;
    // row_id[41] = 9;
    // row_id[42] = 10;
    // row_id[43] = 10;
    // row_id[44] = 10;
    // row_id[45] = 10;
    // row_id[46] = 11;
    // row_id[47] = 11;
    // row_id[48] = 11;
    // row_id[49] = 12;
    // row_id[50] = 12;
    // row_id[51] = 12;
    // row_id[52] = 12;
    // row_id[53] = 12;
    // row_id[54] = 12;
    // row_id[55] = 12;

    int ne = T;
    thrust::universal_vector<thrust::tuple<int32_t,int> > mem(n*ne);
    // //std::cout << mem.size() << std::endl;
    thrust::universal_vector<int> memnnz(n,1);
    for(int i = 0; i < n; i++) {
            thrust::tuple<int32_t,int> init(i, 1);
        for(int j = 0; j < ne; j++) {
                mem[i*ne+j] = init;
                //std::cout << "this is i " << i << "this is j " << j << " " << thrust::get<0>(mem[i*ne+j]) << " " << thrust::get<1>(mem[i*ne+j]) << std::endl;
        }
    }
    std::cout << "finished creating mem\n" << std::endl;
    //thrust::fill(memnnz.begin(), memnnz.end(), 1);

    std::cout << "finished creating memnnz labellsit\n" << std::endl;

    

    double times = 0;
    double timel1 = 0;
    double timel2 = 0;
    double timel2_1 = 0;
    double timel2_2 = 0;
    double timel3 = 0;
    double timepp = 0;
    unsigned int seed;
    thrust::universal_vector<thrust::tuple<int32_t,int>> max_per_row(n);
    for(int i = 0; i < n; i++) {
      max_per_row[i] = thrust::make_tuple(i, 1);
    }
     auto start = std::chrono::high_resolution_clock::now();
    for(int k = 1; k < T; k++) {
        thrust::universal_vector<thrust::tuple<int32_t,int> > labellist(cols, 0);
        seed = k;
        auto starts = std::chrono::high_resolution_clock::now();
        cuco::static_reduction_map<cuco::reduce_add<Value>, custom_key_type, Value> map(cols*2, empty_key_sentinel);
        thrust::for_each(thrust::counting_iterator(0),
                     thrust::counting_iterator(cols),
                    [m = map.get_device_mutable_view(), rowid = thrust::raw_pointer_cast(row_id.data()), mpr = thrust::raw_pointer_cast(max_per_row.data()), c = thrust::raw_pointer_cast(col.data())]
                     __device__ (auto i) mutable {
                        m.insert(cuco::make_pair<custom_key_type, int>(custom_key_type{rowid[i], thrust::get<0>(mpr[c[i]])}, 1));
                    });
        cudaDeviceSynchronize();
        auto ends = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffs = ends- starts;
        times += diffs.count();
        // std::cout << "Time do speaker is "
        //          <<  diffs.count() << " s\n";
        // if(k%mod == 0) {
        //     std::cout << "Speaker labellist \n";
        //     for(int j = 0; j < n; j++) {
        //       std::cout << "j\n";
        //     for(int i = row[j]; i < row[j+1]; i++) {
        //         std::cout << thrust::get<0>(labellist[i]) << ":" << thrust::get<1>(labellist[i]) << " ";
        //     }
        //     std::cout << "\n";
        //   }
        // }
        //exit(0);
        auto startl1 = std::chrono::high_resolution_clock::now();
        // check_duplicates l1(n, row, labellist, row_id);
        // thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols), l1);
        // cudaDeviceSynchronize();
        thrust::universal_vector<int> labellistnnz1(n, 0);
        //thrust::fill(labellistnnz1.begin(), labellistnnz1.end(), 0);
        write_back wb(row, labellistnnz1, labellist, map);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cols*2), wb);
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
        // for(int i = 0; i < n; i++) {
        //   int offset = row[i+1] - row[i];
        //   labellist[row[i]] = thrust::reduce(thrust::device, labellist.begin()+row[i], labellist.begin() + row[i+1], labellist[row[i]], tupleMax());
        // }
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
        thrust::universal_vector<int> labellistnnz(n, 1);
        //thrust::fill(labellistnnz.begin(), labellistnnz.end(), 1);
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

        // for(int i = 0; i < n; i++) {
        //   std::cout << thrust::get<0>(max_per_row[i]) << " " << thrust::get<1>(max_per_row[i]) << "\n";
        // }
        update_memory l3(n, row, mem, labellist, memnnz, ne, max_per_row);
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), l3);
        cudaDeviceSynchronize();
        // for(int i = 0; i < n; i++) {
        //   std::cout << thrust::get<0>(max_per_row[i]) << " " << thrust::get<1>(max_per_row[i]) << "\n";
        // }
        auto endl3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffl3 = endl3 - startl3;
        timel3 += diffl3.count();
        //exit(0);
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
        //  auto startllf = std::chrono::high_resolution_clock::now();
        // thrust::fill(labellist.begin(), labellist.end(), 0);
        //  auto endllf = std::chrono::high_resolution_clock::now();
        //   std::chrono::duration<double> diffllf = endllf - startllf;
        //   timellf += diffllf.count();
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

        std::cout << "time for speaker is "
                  <<  times << " s\n";
        std::cout << "time for listerner 1 is "
                  <<  timel1 << " s\n";
        std::cout << "time for listerner 2 is "
                  <<  timel2 << " s\n";
        std::cout << "time for listerner 2_1 is "
                  <<  timel2_1 << " s\n";
        std::cout << "time for listerner 2_2 is "
                  <<  timel2_2 << " s\n";
        std::cout << "time for listerner 3 is "
                  <<  timel3 << " s\n";
        std::cout << "time for post is "
                  <<  diffpp.count() << " s\n";
    //   for(int i = 0; i < n; i++) {
    //     int count = 0;
    //     std::cout << "Vertex " << i << " is in communities ";
    //     for(int j = 0; j < memnnz[i]; j++) {
    //         if(thrust::get<0>(mem[i*T+j]) != -1) {
    //             std::cout << thrust::get<0>(mem[i*T+j]) << " ";
    //             count++;
    //         }
    //     }
    //     if(count > 1)
    //         std::cout << "overlap\n";
    //     else
    //         std::cout << " only community\n";
    // }
    return 0;
}

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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

template <typename vertex_t, typename edge_t>
void SLPA_reference(edge_t const* offsets,
                              vertex_t const* indices,  // each adjacency list should be sorted
                              vertex_t num_vertices,
                            )
{
  if (num_vertices == 0) { return; }

  return;
}

struct SLPA_Usecase {
  double vertex_subset_ratio{0.0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_SLPA
  : public ::testing::TestWithParam<std::tuple<SLPA_Usecase, input_usecase_t>> {
 public:
  Tests_SLPA() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(
    std::tuple<SLPA_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber = true;

    using weight_t = float;

    auto [triangle_count_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResClock hr_clock{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto graph_view = graph.view();

    std::optional<std::vector<vertex_t>> h_vertices{std::nullopt};
    if (triangle_count_usecase.vertex_subset_ratio < 1.0) {
      std::default_random_engine generator{};
      std::uniform_real_distribution<double> distribution{0.0, 1.0};
      h_vertices = std::vector<vertex_t>(graph_view.number_of_vertices());
      std::iota((*h_vertices).begin(),
                (*h_vertices).end(),
                graph_view.local_vertex_partition_range_first());
      (*h_vertices)
        .erase(std::remove_if((*h_vertices).begin(),
                              (*h_vertices).end(),
                              [&generator, &distribution, triangle_count_usecase](auto) {
                                return distribution(generator) >=
                                       triangle_count_usecase.vertex_subset_ratio;
                              }),
               (*h_vertices).end());
    }

    auto d_vertices = h_vertices ? std::make_optional<rmm::device_uvector<vertex_t>>(
                                     (*h_vertices).size(), handle.get_stream())
                                 : std::nullopt;
    if (d_vertices) {
      raft::update_device(
        (*d_vertices).data(), (*h_vertices).data(), (*h_vertices).size(), handle.get_stream());
    }

    rmm::device_uvector<edge_t> d_triangle_counts(
      d_vertices ? (*d_vertices).size() : graph_view.number_of_vertices(), handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    cugraph::triangle_count<vertex_t, edge_t, weight_t, false>(
      handle,
      graph_view,
      d_vertices ? std::make_optional<raft::device_span<vertex_t const>>((*d_vertices).begin(),
                                                                         (*d_vertices).end())
                 : std::nullopt,
      raft::device_span<edge_t>(d_triangle_counts.begin(), d_triangle_counts.end()),
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "Triangle count took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (triangle_count_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> unrenumbered_graph(handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, false, false, false, true);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      std::vector<edge_t> h_offsets(unrenumbered_graph_view.number_of_vertices() + 1);
      std::vector<vertex_t> h_indices(unrenumbered_graph_view.number_of_edges());
      raft::update_host(h_offsets.data(),
                        unrenumbered_graph_view.local_edge_partition_view().offsets(),
                        unrenumbered_graph_view.number_of_vertices() + 1,
                        handle.get_stream());
      raft::update_host(h_indices.data(),
                        unrenumbered_graph_view.local_edge_partition_view().indices(),
                        unrenumbered_graph_view.number_of_edges(),
                        handle.get_stream());

      handle.sync_stream();

      std::vector<edge_t> h_reference_triangle_counts(unrenumbered_graph_view.number_of_vertices());

      triangle_count_reference(h_offsets.data(),
                               h_indices.data(),
                               unrenumbered_graph_view.number_of_vertices(),
                               h_reference_triangle_counts.data());

      std::vector<vertex_t> h_cugraph_vertices(d_triangle_counts.size());
      if (d_vertices) {
        if (renumber) {
          cugraph::unrenumber_local_int_vertices(handle,
                                                 (*d_vertices).data(),
                                                 (*d_vertices).size(),
                                                 (*d_renumber_map_labels).data(),
                                                 vertex_t{0},
                                                 graph_view.number_of_vertices(),
                                                 true);
        }
        raft::update_host(h_cugraph_vertices.data(),
                          (*d_vertices).data(),
                          (*d_vertices).size(),
                          handle.get_stream());
      } else {
        if (renumber) {
          raft::update_host(h_cugraph_vertices.data(),
                            (*d_renumber_map_labels).data(),
                            (*d_renumber_map_labels).size(),
                            handle.get_stream());
        } else {
          std::iota(h_cugraph_vertices.begin(), h_cugraph_vertices.end(), vertex_t{0});
        }
      }
      std::vector<edge_t> h_cugraph_triangle_counts(d_triangle_counts.size());
      raft::update_host(h_cugraph_triangle_counts.data(),
                        d_triangle_counts.data(),
                        d_triangle_counts.size(),
                        handle.get_stream());

      handle.sync_stream();

      for (size_t i = 0; i < h_cugraph_vertices.size(); ++i) {
        auto v     = h_cugraph_vertices[i];
        auto count = h_cugraph_triangle_counts[i];
        ASSERT_TRUE(count == h_reference_triangle_counts[v])
          << "Triangle count values do not match with the reference values.";
      }
    }
  }
};

using Tests_SLPA_File = Tests_SLPA<cugraph::test::File_Usecase>;
using Tests_SLPA_Rmat = Tests_SLPA<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_SLPA_File, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SLPA_Rmat, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SLPA_File, CheckInt32Int64)
{
  run_current_test<int32_t, int64_t>(override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SLPA_Rmat, CheckInt64Int64)
{
  run_current_test<int64_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SLPA_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(SLPA_Usecase{0.1}, SLPA_Usecase{1.0}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_SLPA_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(SLPA_Usecase{0.1}, SLPA_Usecase{1.0}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_SLPA_File,
  ::testing::Combine(
    // disable correctness checks
    ::testing::Values(SLPA_Usecase{0.1, false}, SLPA_Usecase{1.0, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_SLPA_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(SLPA_Usecase{0.1, false}, SLPA_Usecase{1.0, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

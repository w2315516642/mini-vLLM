import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from minivllm.configs import ParallelConfig
from minivllm.engine import ray_utils


def make_node(ip, num_gpus=2, *, head=False, alive=True):
    # Ray's internal head marker can precede the actual node resource.
    resources = {"node:__internal_head__": 1.0} if head else {}
    resources[f"node:{ip}"] = 1.0
    if num_gpus:
        resources["GPU"] = float(num_gpus)
    return {
        "Alive": alive,
        "NodeManagerAddress": ip,
        "Resources": resources,
    }


class RayClusterInitializationTest(unittest.TestCase):
    def setUp(self):
        self.ray = SimpleNamespace(init=Mock(), nodes=Mock(return_value=[]))
        ray_patch = patch.object(ray_utils, "ray", self.ray)
        ray_patch.start()
        self.addCleanup(ray_patch.stop)
        port_patch = patch.object(ray_utils.random, "randint", return_value=19493)
        port_patch.start()
        self.addCleanup(port_patch.stop)

    def initialize(self, nodes, tensor_parallel_size=2):
        self.ray.nodes.return_value = nodes
        config = ParallelConfig(1, tensor_parallel_size, worker_use_ray=True)
        return ray_utils.initialize_cluster(config, ray_address="test-cluster")

    def test_head_resource_is_not_a_rendezvous_hostname(self):
        method, devices = self.initialize([make_node("10.0.0.10", head=True)])

        self.assertEqual(method, "tcp://10.0.0.10:19493")
        self.assertEqual(devices, [[
            (0, "node:10.0.0.10", 0), (1, "node:10.0.0.10", 1),
        ]])
        self.ray.init.assert_called_once_with(address="test-cluster")

    def test_resource_iteration_order_does_not_change_the_node_address(self):
        node = make_node("10.0.0.10", head=True)
        node["Resources"] = dict(reversed(list(node["Resources"].items())))
        method, devices = self.initialize([node])

        self.assertEqual(method, "tcp://10.0.0.10:19493")
        self.assertTrue(all(device[1] == "node:10.0.0.10" for device in devices[0]))

    def test_head_marker_does_not_inflate_gpu_capacity(self):
        with self.assertRaisesRegex(ValueError, "required GPUs exceeds"):
            self.initialize([make_node("10.0.0.10", head=True)], 4)

    def test_two_nodes_assign_ranks_to_two_distinct_addresses(self):
        method, devices = self.initialize([
            make_node("10.0.0.10", head=True),
            make_node("10.0.0.11"),
        ], 4)

        self.assertEqual(method, "tcp://10.0.0.10:19493")
        self.assertEqual(devices, [[
            (0, "node:10.0.0.10", 0), (1, "node:10.0.0.10", 1),
            (2, "node:10.0.0.11", 0), (3, "node:10.0.0.11", 1),
        ]])

    def test_cpu_only_and_dead_nodes_are_skipped(self):
        method, devices = self.initialize([
            make_node("10.0.0.1", 0, head=True),
            make_node("10.0.0.2", alive=False),
            make_node("10.0.0.10"),
        ])

        self.assertEqual(method, "tcp://10.0.0.10:19493")
        self.assertEqual(len(devices[0]), 2)

    def test_no_gpu_nodes_reports_an_explicit_error(self):
        for nodes in ([], [make_node("10.0.0.1", 0, head=True)]):
            with self.subTest(nodes=nodes):
                with self.assertRaisesRegex(ValueError, "No alive Ray nodes with GPUs"):
                    self.initialize(nodes)

    def test_nonuniform_gpu_counts_remain_unsupported(self):
        with self.assertRaisesRegex(ValueError, "not uniform"):
            self.initialize([make_node("10.0.0.10"), make_node("10.0.0.11", 4)])

    def test_other_node_labels_do_not_inflate_capacity(self):
        node = make_node("10.0.0.10")
        node["Resources"]["node:custom-label"] = 1.0
        with self.assertRaisesRegex(ValueError, "required GPUs exceeds"):
            self.initialize([node], 4)

    def test_local_single_gpu_does_not_initialize_ray(self):
        config = ParallelConfig(1, 1, worker_use_ray=False)
        method, devices = ray_utils.initialize_cluster(config)

        self.assertEqual(method, "tcp://localhost:19493")
        self.assertEqual(devices, [[(0, None, 0)]])
        self.ray.init.assert_not_called()
        self.ray.nodes.assert_not_called()


if __name__ == "__main__":
    unittest.main()

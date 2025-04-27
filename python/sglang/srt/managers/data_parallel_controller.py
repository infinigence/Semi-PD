# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import logging
import multiprocessing as mp
import os
import signal
import threading
from enum import Enum, auto
from typing import List, Union

import psutil
import setproctitle
import zmq

from sglang.semi_pd.utils import (
    DECODE_ENGINE_SM_PERCENTILE,
    PREFILL_ENGINE_SM_PERCENTILE,
    AggregatedSocket,
    InstanceRole,
)
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.semi_pd_scheduler import run_standalone_scheduler_process
from sglang.srt.server_args import PortArgs, SemiPDPortArgs, ServerArgs
from sglang.srt.utils import bind_port, configure_logger, get_zmq_socket
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ) -> None:
        """
        Semi-PD:
        - increase the number of zmq io threads when semi-pd is enabled
        - use AggregatedSocket for prefill and decode
        """
        # Parse args
        self.max_total_num_tokens = None
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        # Init inter-process communication
        if server_args.enable_semi_pd:
            # Semi-PD
            self.context = zmq.Context(1 + server_args.dp_size * 3)
        else:
            self.context = zmq.Context(1 + server_args.dp_size)

        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Launch data parallel workers
        self.scheduler_procs = []
        self.workers: List[Union[zmq.Socket, AggregatedSocket]] = []

        if not server_args.enable_dp_attention:
            dp_port_args = self.launch_dp_schedulers(server_args, port_args)
        else:
            dp_port_args = self.launch_dp_attention_schedulers(server_args, port_args)

        # Only node rank 0 runs the real data parallel controller that dispatches the requests.
        if server_args.node_rank == 0:
            if server_args.enable_semi_pd:
                # Semi-PD
                for dp_rank in range(server_args.dp_size):
                    pa = dp_port_args[dp_rank]
                    assert isinstance(pa, SemiPDPortArgs)
                    prefill_socket = get_zmq_socket(
                        self.context,
                        zmq.PUSH,
                        pa.p_scheduler_input_ipc_name,
                        False,
                    )
                    decode_socket = get_zmq_socket(
                        self.context,
                        zmq.PUSH,
                        pa.d_scheduler_input_ipc_name,
                        False,
                    )
                    self.workers.append(
                        # Decode first, for better performance.
                        AggregatedSocket([decode_socket, prefill_socket])
                    )
            else:
                for dp_rank in range(server_args.dp_size):
                    self.workers.append(
                        get_zmq_socket(
                            self.context,
                            zmq.PUSH,
                            dp_port_args[dp_rank].scheduler_input_ipc_name,
                            True,
                        )
                    )

        self.max_req_input_len = None

    def launch_dp_schedulers(self, server_args: ServerArgs, port_args: PortArgs):
        """
        Semi-PD:
        - use SemiPDPortArgs when semi-pd is enabled
        """

        base_gpu_id = 0

        threads = []
        sockets = []
        dp_port_args = []
        for dp_rank in range(server_args.dp_size):
            if server_args.enable_semi_pd:
                tmp_port_args = SemiPDPortArgs.init_new(server_args)
            else:
                tmp_port_args = PortArgs.init_new(server_args)

            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            dp_port_args.append(tmp_port_args)

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            if server_args.enable_semi_pd:
                assert isinstance(tmp_port_args, SemiPDPortArgs)
                sockets.append(bind_port(tmp_port_args.s_nccl_port))
                sockets.append(bind_port(tmp_port_args.p_nccl_port))
                sockets.append(bind_port(tmp_port_args.d_nccl_port))
            else:
                assert isinstance(tmp_port_args, PortArgs)
                sockets.append(bind_port(tmp_port_args.nccl_port))

            # Create a thread for each worker
            target = self.launch_tensor_parallel_group
            if self.server_args.enable_semi_pd:
                target = self.launch_semi_pd_tensor_parallel_group

            thread = threading.Thread(
                target=target,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank),
            )
            threads.append(thread)
            base_gpu_id += server_args.tp_size * server_args.gpu_id_step

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return dp_port_args

    def launch_dp_attention_schedulers(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ) -> List[Union[PortArgs, SemiPDPortArgs]]:

        assert isinstance(port_args, PortArgs)

        if server_args.enable_semi_pd:
            tmp_port_args = SemiPDPortArgs.init_new(server_args)
            assert tmp_port_args.tokenizer_ipc_name == port_args.tokenizer_ipc_name
            assert tmp_port_args.detokenizer_ipc_name == port_args.detokenizer_ipc_name

            self.launch_semi_pd_tensor_parallel_group(
                server_args, tmp_port_args, 0, None
            )
            dp_port_args = []
            for dp_rank in range(server_args.dp_size):
                dp_port_args.append(SemiPDPortArgs.init_new(server_args, dp_rank))
            return dp_port_args
        else:
            self.launch_tensor_parallel_group(server_args, port_args, 0, None)
            dp_port_args = []
            for dp_rank in range(server_args.dp_size):
                dp_port_args.append(PortArgs.init_new(server_args, dp_rank))
            return dp_port_args

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
    ):
        from sglang.srt.managers.scheduler import run_scheduler_process

        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        # Launch tensor parallel scheduler processes
        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        for tp_rank in tp_rank_range:
            rank_port_args = port_args

            if server_args.enable_dp_attention:
                # dp attention has different sharding logic
                _, _, dp_rank = compute_dp_attention_world_info(
                    server_args.enable_dp_attention,
                    tp_rank,
                    server_args.tp_size,
                    server_args.dp_size,
                )
                # compute zmq ports for this dp rank
                rank_port_args = PortArgs.init_new(server_args, dp_rank)
                # Data parallelism resues the tensor parallelism group,
                # so all dp ranks should use the same nccl port.
                rank_port_args.nccl_port = port_args.nccl_port

            reader, writer = mp.Pipe(duplex=False)
            gpu_id = (
                server_args.base_gpu_id
                + base_gpu_id
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, rank_port_args, gpu_id, tp_rank, dp_rank, writer),
            )
            proc.start()
            self.scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def get_semi_pd_dp_attention_world_info(
        self,
        server_args: ServerArgs,
        port_args: SemiPDPortArgs,
        tp_rank: int,
        dp_rank: int,
    ):
        # dp attention has different sharding logic
        _, _, dp_rank = compute_dp_attention_world_info(
            server_args.enable_dp_attention,
            tp_rank,
            server_args.tp_size,
            server_args.dp_size,
        )
        # compute zmq ports for this dp rank
        rank_port_args = SemiPDPortArgs.init_new(server_args, dp_rank)

        # Data parallelism resues the tensor parallelism group,
        # so all dp ranks should use the same nccl port.
        rank_port_args.s_nccl_port = port_args.s_nccl_port
        rank_port_args.p_nccl_port = port_args.p_nccl_port
        rank_port_args.d_nccl_port = port_args.d_nccl_port

        assert rank_port_args.tokenizer_ipc_name == port_args.tokenizer_ipc_name
        assert rank_port_args.detokenizer_ipc_name == port_args.detokenizer_ipc_name

        return rank_port_args, dp_rank

    def launch_semi_pd_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: SemiPDPortArgs,
        base_gpu_id: int,
        dp_rank: int,
    ):
        from sglang.srt.managers.semi_pd_scheduler import run_scheduler_process

        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        # Launch tensor parallel scheduler processes
        p_scheduler_pipe_readers = []
        d_scheduler_pipe_readers = []
        standalone_scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        p_ipc_info_queues: List[mp.Queue] = [
            mp.Queue() for _ in range(tp_size_per_node)
        ]
        d_ipc_info_queues: List[mp.Queue] = [
            mp.Queue() for _ in range(tp_size_per_node)
        ]

        for tp_rank in tp_rank_range:
            rank_port_args = port_args
            if server_args.enable_dp_attention:
                rank_port_args, dp_rank = self.get_semi_pd_dp_attention_world_info(
                    server_args, port_args, tp_rank, dp_rank
                )

            reader_standalone, writer_standalone = mp.Pipe(duplex=False)
            gpu_id = (
                server_args.base_gpu_id
                + base_gpu_id
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            queue_idx = tp_rank % tp_size_per_node
            p_ipc_info_queue = p_ipc_info_queues[queue_idx]
            d_ipc_info_queue = d_ipc_info_queues[queue_idx]
            proc = mp.Process(
                target=run_standalone_scheduler_process,
                args=(
                    server_args,
                    rank_port_args,
                    gpu_id,
                    tp_rank,
                    dp_rank,
                    writer_standalone,
                    False,
                    p_ipc_info_queue,
                    d_ipc_info_queue,
                ),
            )
            proc.start()
            self.scheduler_procs.append(proc)
            standalone_scheduler_pipe_readers.append(reader_standalone)

        # Wait for model to finish loading
        scheduler_info = []
        tp_rank_base = tp_size_per_node * server_args.node_rank
        max_total_num_tokens = None
        for i, reader in enumerate(standalone_scheduler_pipe_readers):
            logger.info(
                f"Waiting for standalone scheduler {tp_rank_base + i} to be ready"
            )
            data = reader.recv()
            assert data["status"] == "ready"
            # Get max_total_num_tokens from standalone schedulers
            if i > 0:
                assert data["max_total_num_tokens"] == max_total_num_tokens
            max_total_num_tokens = data["max_total_num_tokens"]

        # P & D schedulers use the same max_total_num_tokens from the standalone scheduler.
        assert max_total_num_tokens is not None
        server_args.max_total_tokens = max_total_num_tokens

        # Init P & D schedulers.
        for tp_rank in tp_rank_range:
            rank_port_args = port_args
            if server_args.enable_dp_attention:
                rank_port_args, dp_rank = self.get_semi_pd_dp_attention_world_info(
                    server_args, port_args, tp_rank, dp_rank
                )

            gpu_id = (
                server_args.base_gpu_id
                + base_gpu_id
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            queue_idx = tp_rank % tp_size_per_node
            p_ipc_info_queue = p_ipc_info_queues[queue_idx]
            d_ipc_info_queue = d_ipc_info_queues[queue_idx]

            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                DECODE_ENGINE_SM_PERCENTILE
            )
            logger.info(
                f"Launch D instance TP {tp_rank} with {os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}% SMs"
            )
            reader_decode, writer_decode = mp.Pipe(duplex=False)
            proc = mp.Process(
                target=run_scheduler_process,
                args=(
                    server_args,
                    rank_port_args,
                    gpu_id,
                    tp_rank,
                    dp_rank,
                    writer_decode,
                    d_ipc_info_queue,
                    True,
                    InstanceRole.DECODE,
                ),
            )
            proc.start()
            self.scheduler_procs.append(proc)
            d_scheduler_pipe_readers.append(reader_decode)

            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                PREFILL_ENGINE_SM_PERCENTILE
            )
            logger.info(
                f"Launch P instance TP {tp_rank} with {os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}% SMs"
            )
            reader_prefill, writer_prefill = mp.Pipe(duplex=False)
            proc_shared = mp.Process(
                target=run_scheduler_process,
                args=(
                    server_args,
                    rank_port_args,
                    gpu_id,
                    tp_rank,
                    dp_rank,
                    writer_prefill,
                    p_ipc_info_queue,
                    True,
                    InstanceRole.PREFILL,
                ),
            )

            proc_shared.start()
            self.scheduler_procs.append(proc_shared)
            p_scheduler_pipe_readers.append(reader_prefill)

        for i, reader in enumerate(p_scheduler_pipe_readers):
            logger.info(f"Waiting for P instance {tp_rank_base + i} to be ready")
            data = reader.recv()
            assert data["status"] == "ready"
            scheduler_info.append(data)

        for i, reader in enumerate(d_scheduler_pipe_readers):
            logger.info(f"Waiting for D instance {tp_rank_base + i} to be ready")
            data = reader.recv()
            assert data["status"] == "ready"
            scheduler_info.append(data)

        logger.info("All schedulers are ready")

        self.max_total_num_tokens = max_total_num_tokens
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def round_robin_scheduler(self, req):
        self.workers[self.round_robin_counter].send_pyobj(req)
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)

    def shortest_queue_scheduler(self, input_requests):
        raise NotImplementedError()

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

                if isinstance(
                    recv_req,
                    (
                        TokenizedGenerateReqInput,
                        TokenizedEmbeddingReqInput,
                    ),
                ):
                    self.dispatching(recv_req)
                else:
                    # Send other control messages to first worker of tp group
                    for worker in self.workers[:: self.server_args.tp_size]:
                        worker.send_pyobj(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        controller = DataParallelController(server_args, port_args)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)

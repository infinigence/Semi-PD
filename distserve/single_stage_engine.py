import time, copy
from typing import Callable, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import asyncio
import os
import numpy as np
from enum import Enum
import contextvars
from contextlib import contextmanager

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

import torch

from distserve.logger import init_logger
from distserve.config import (
    ModelConfig,
    ParallelConfig,
    CacheConfig,
)
from distserve.request import (
    Request, 
    BatchedRequests,
    MigratingRequest
)
from distserve.utils import Counter, cudaMemoryIpcHandle, Stage
from distserve.lifetime import LifetimeEvent, LifetimeEventType
from distserve.tokenizer import get_tokenizer
from distserve.block_manager import BlockManager
from distserve.worker import ParaWorker, ENABLE_DYNAMIC_SWITCH, ENABLE_MPS
from distserve.context_stage_scheduler import ContextStageSchedConfig, ContextStageScheduler, get_context_stage_scheduler
from distserve.decoding_stage_scheduler import DecodingStageSchedConfig, DecodingStageScheduler, get_decoding_stage_scheduler

logger = init_logger(__name__)

# Sleep for this many seconds when there is no request in ContextStageLLMEngine.step()
# We need to sleep for a while because the whole program is a asyncio-based,
# event driven, single thread program. We save some CPU time for other coroutines.
SLEEP_WHEN_CONTEXT_NO_REQUEST = 0.003

# Sleep for this many seconds when there is no request in DecodingStageLLMEngine.step()
SLEEP_WHEN_DECODING_NO_REQUEST = 0.003

# Sleep for this many seconds in each event loop, useful for debugging
SLEEP_IN_EACH_EVENT_LOOP = 0

# Print engine status every this many seconds
PRINT_STATUS_INTERVAL = 1

SLEEP_FOR_MPS_RESCHEDULE = 0.5

# ENABLE_DYNAMIC_SWITCH = bool(os.getenv("ENABLE_DYNAMIC_SWITCH", 0))
# logger.info("ENABLE_DYNAMIC_SWITCH=%s", ENABLE_DYNAMIC_SWITCH)

# 创建一个上下文变量
WorkersManager = contextvars.ContextVar('workers_manager', default=0)

@contextmanager
def WorkersContext(new_value):
    token = WorkersManager.set(new_value)
    try:
        yield WorkersManager
    finally:
        WorkersManager.reset(token)

class StepOutput:
    """The output of request in one step of inference.
    It contains the information of corresponding request and the generated tokens until this step.
    """

    def __init__(self, request: Request, new_token: str, new_token_id: int):
        self.request = request
        self.request_id = request.request_id
        self.prompt = request.prompt
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = request.is_finished

    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finished})"
        )
class SwitchSemaphore(Enum):
    """
    The type of an event in a request's lifetime
    """
    REJECT = "REJECT"
    START_INIT = "START_INIT",
    # INITIALIZING = "INITIALIZING",
    END_INIT = "END_INIT",
    ACCEPT = "ACCEPT",

    def __str__(self) -> str:
        return self.value


TTFT_SLO  =  int(
    os.getenv("TTFT_SLO", 500))  # noqa
TPOT_SLO  =  int(
    os.getenv("TPOT_SLO", 100))  # noqa

logger.info("\033[1;32;40mTTFT_SLO=%s TPOT_SLO=%s\033[0m", TTFT_SLO, TPOT_SLO)

class DynamicPartitioner():
    def __init__(self, tpot_window: List, ttft_window: List, prefill_sm_percentile: int, decode_sm_percentile: int):
        self.tpot_window = tpot_window
        self.ttft_window = ttft_window
        self.prefill_cur_sm_percentile = prefill_sm_percentile
        self.decode_cur_sm_percentile = decode_sm_percentile
        self.prefill_future_sm_percentile = prefill_sm_percentile
        self.decode_future_sm_percentile = decode_sm_percentile
        
        # hyper params
        self.window_size = 400
        self.tpot_scale = 1.1
        
        # SLA
        self.ttft_slo = TTFT_SLO
        self.tpot_slo = TPOT_SLO
        
        # semaphore
        self.do_prefill_replan = False
        self.prefill_switch_status = None

        
        self.do_decode_replan = False
        self.decode_switch_status = None
    
    def update_prefill_status(self, status):
        self.prefill_switch_status = status
    
    def update_decode_status(self, status):
        self.decode_switch_status = status
    
    def update_decode_info(self, tpot):
        if not self.do_prefill_replan and self.prefill_switch_status == SwitchSemaphore.REJECT:
            self.tpot_window.append(tpot)

    def update_prefill_info(self, prefill_info):
        if not self.do_prefill_replan and self.prefill_switch_status == SwitchSemaphore.REJECT:
            self.ttft_window.append(prefill_info)

    def check_switch_decode(self):
        return self.do_decode_replan
    
    def check_switch_prefill(self):
        return self.do_prefill_replan
    
    def get_decode_predicted_sm(self):
        return self.decode_future_sm_percentile
    
    def get_prefill_predicted_sm(self):
        return self.prefill_future_sm_percentile

    def replan(self):
        def _scale_tpot(sm_percentile):
            if sm_percentile == 100:
                return 1
            inv_sm_p = 100 / sm_percentile
            scale =  0.6099164272415375 * inv_sm_p + 0.41530605763705736
            return scale
        
        def _scale_waiting(sm_percentile):
            if sm_percentile == 100:
                return 1
            inv_sm_p = 100 / sm_percentile
            scale = 1.4046465674985547 * inv_sm_p ** 2 + -3.15935463280878 * inv_sm_p + 2.8734660485396604
            return scale
            
        def _scale_latency(sm_percentile):
            if sm_percentile == 100:
                return 1
            inv_sm_p = 100 / sm_percentile
            scale =  0.732435017003385 * inv_sm_p + 0.3185253290850649
            return scale
        
        # Switch in progress
        if self.do_prefill_replan or self.do_decode_replan:
            return False
        
        # every (windows size amount) reqs check once
        if not self.tpot_window.__len__() >= self.window_size:
            return False
        
        p90_tpot = np.percentile(self.tpot_window, 90)
        # clean tpot
        self.tpot_window.clear()

        if self.ttft_window.__len__() != 0:
            ttft_window = sorted(self.ttft_window, key = lambda x:x[0])
            p90_idx = int(len(ttft_window) * 0.9 / 1 + 1)
            p90_ttft, p90_latency, p90_waiting = ttft_window[p90_idx]
            self.ttft_window.clear()
        else:
            p90_ttft, p90_latency, p90_waiting = (-1, -1, -1)
        
        cond_tpot = p90_tpot * self.tpot_scale <= self.tpot_slo
        cond_ttft = p90_ttft <= self.ttft_slo
        
        logger.info("\033[1;32;40m p90_TTFT:%f p90_TPOT %f\033[0m",  p90_ttft, p90_tpot)
        
        # Since both cannot be satisfied, just maximize mfu
        if not cond_tpot and not cond_ttft and (self.prefill_cur_sm_percentile != 100 and self.decode_cur_sm_percentile !=100):
            self.decode_future_sm_percentile = 100
            self.prefill_future_sm_percentile = 100
            self.do_decode_replan = True
            self.do_prefill_replan = True
            return True
        
        interval = 2
        adjust_count = 0
        max_step = 3
        # First to check tpot constraint
        if not cond_tpot and cond_ttft:
            # adjust decode sm
            if self.decode_cur_sm_percentile < 100:
                predicted_sm_percentile = self.decode_cur_sm_percentile
                base_tpot_scale = _scale_tpot(self.decode_cur_sm_percentile)
                while True and adjust_count < max_step:
                    # adjust to 100
                    predicted_sm_percentile += interval
                    adjust_count +=1
                    if predicted_sm_percentile >= 100:
                        predicted_sm_percentile = 100
                    future_tpot_scale = _scale_tpot(predicted_sm_percentile)
                    future_tpot = future_tpot_scale / base_tpot_scale * p90_tpot
                    # shortest step
                    if future_tpot <= self.tpot_slo:
                        break
                    if predicted_sm_percentile == 100:
                        logger.info("meet the upper bound, adjustment limited")
                        break
                self.decode_future_sm_percentile = predicted_sm_percentile
                self.do_decode_replan = True
                return True
            # slo_ttft > ttft
            else:
                predicted_sm_percentile = self.prefill_cur_sm_percentile
                while True and adjust_count < max_step:
                    # adjust to 40(hard lower bound)
                    base_latency_scale = _scale_latency(self.prefill_cur_sm_percentile)
                    base_waiting_scale = _scale_waiting(self.prefill_cur_sm_percentile)
                    predicted_sm_percentile -= interval
                    adjust_count +=1
                    if predicted_sm_percentile <= 80:
                        predicted_sm_percentile = 80
                    future_waiting_scale = _scale_waiting(predicted_sm_percentile)
                    future_latency_scale = _scale_latency(predicted_sm_percentile)
                    future_ttft = future_waiting_scale / base_waiting_scale * p90_waiting + future_latency_scale / base_latency_scale * p90_latency
                    # longest step
                    if future_ttft > self.ttft_slo:
                        predicted_sm_percentile += interval
                        break
                    if predicted_sm_percentile == 80:
                        logger.info("meet the lower bound, adjustment limited")
                        break
                self.prefill_future_sm_percentile = predicted_sm_percentile
                self.do_prefill_replan = True
                return True
        
        if cond_tpot and not cond_ttft:
            if self.prefill_cur_sm_percentile < 100:
                predicted_sm_percentile = self.prefill_cur_sm_percentile
                while True and adjust_count < max_step:
                    base_latency_scale = _scale_latency(self.prefill_cur_sm_percentile)
                    base_waiting_scale = _scale_waiting(self.prefill_cur_sm_percentile)
                    # adjust to 100
                    predicted_sm_percentile += interval
                    adjust_count +=1
                    if predicted_sm_percentile >= 100:
                        predicted_sm_percentile = 100
                    future_waiting_scale = _scale_waiting(predicted_sm_percentile)
                    future_latency_scale = _scale_latency(predicted_sm_percentile)
                    future_ttft = future_waiting_scale / base_waiting_scale * p90_waiting + future_latency_scale / base_latency_scale * p90_latency
                    # shortest step
                    if future_ttft <= self.ttft_slo:
                        break
                    if predicted_sm_percentile == 100:
                        logger.info("meet the upper bound, adjustment limited")
                        break
                self.prefill_future_sm_percentile = predicted_sm_percentile
                self.do_prefill_replan = True
                return True
            else:
                predicted_sm_percentile = self.decode_cur_sm_percentile
                while True and adjust_count < max_step:
                    base_tpot_scale = _scale_tpot(self.decode_cur_sm_percentile)
                    # adjust to 60(hard lower bound)
                    predicted_sm_percentile -= interval
                    adjust_count +=1
                    if predicted_sm_percentile <= 80:
                        predicted_sm_percentile = 80
                    future_tpot_scale = _scale_tpot(predicted_sm_percentile)
                    future_tpot = future_tpot_scale / base_tpot_scale * p90_tpot
                    # longest step
                    if future_tpot > self.tpot_slo:
                        predicted_sm_percentile += interval
                        break
                    if predicted_sm_percentile == 80:
                        logger.info("meet the lower bound, adjustment limited")
                        break
                self.decode_future_sm_percentile = predicted_sm_percentile
                self.do_decode_replan = True
                return True
    
class SingleStageLLMEngine(ABC):
    """
    SingleStageLLMEngine: An LLMEngine that runs either the context stage or the decoding stage.
    
    This class is the base class for ContextStageLLMEngine and DecodingStageLLMEngine.
    """
    @abstractmethod
    def _get_scheduler(self) -> ContextStageScheduler | DecodingStageScheduler:
        raise NotImplementedError()
    
    def _free_request_resources(self, request_id: int) -> None:
        self.block_manager.free_blocks(request_id)
        self._remote_call_all_workers_async("clear_request_resource", request_id)
    
    def __init__(
        self,
        stage: Stage,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig | DecodingStageSchedConfig,
        placement_groups: List[PlacementGroup],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],   # The LLMEngine's callback function when a new StepOutput of a particular request is generated
        engine_on_new_lifetime_event_callback: Optional[Callable[[int, LifetimeEvent, bool], None]] = None,   # The LLMEngine's callback function when a new LifetimeEvent of a particular request is generated
        peer_ids = None
    ):
        self.stage = stage
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.sched_config = sched_config
        self.engine_on_new_step_output_callback = engine_on_new_step_output_callback
        self.engine_on_new_lifetime_event_callback = engine_on_new_lifetime_event_callback

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )

        self.placement_groups = placement_groups
        
        self.peer_ids = peer_ids
        
        # workers[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.workers = []
    
        self.workers_back_up = []
        self.workers_back_up_1 = []
        
        self.worker_index = 0
    
        self.persistant_worker = None
        
        self.switch_semaphore = SwitchSemaphore.REJECT
        self.switch_percentile = 100
        self.sm_percentile = 100
        self.semaphore_queue = []
        self.step_count = 1
        self.ttft_slo = 500
        self.tpot_slo = 100
        self.dyn_partitioner : DynamicPartitioner = None
    
    def set_dyn_partitioner(self, dyn_partitioner):
        self.dyn_partitioner = dyn_partitioner
    
    def insert_semaphore(self, percentile):
        self.semaphore_queue.append(percentile)
         
    async def set_persistent_worker(self):
        self.persistant_worker = self.workers
        self.workers = self.workers_back_up
        self.workers_back_up = self.workers_back_up_1
        
    async def worker_switch(self):
        logger.info("\033[1;32;40mWorker switched, semaphore=%s\033[0m",self.switch_semaphore.__str__())
        self.worker_index ^= 1
        tmp_worker = self.workers
        self.workers = self.workers_back_up
        self.workers_back_up = tmp_worker
    
    async def initialize(self, bypass_block_init=False, bypass_worker_init=False, bypass_weigit_init=False):
        """Initialize workers, load models and initialize k/v cache
        
        We seperate this function from __init__ because we want to run it in an async way
        to enable parallel initialization between Engines.
        """
        self.bypass_block_init = bypass_block_init
        self.bypass_worker_init = bypass_worker_init
        
        if not bypass_worker_init:
            logger.info(f"Initializing {self.stage.name} workers")
            await self._init_workers()
        
        logger.info(f"Initializing {self.stage.name} models")
        await self._init_model(bypass_weigit_init=bypass_weigit_init)
        
        logger.info(f"Initializing {self.stage.name} kvcaches")
        self.num_gpu_blocks, self.num_cpu_blocks = await self._init_kvcache(bypass_block_init)

        if not bypass_block_init:
            self.block_manager = BlockManager(
                self.stage,
                self.num_gpu_blocks,
                self.num_cpu_blocks,
                self.model_config,
                self.parallel_config,
                self.cache_config,
                self._remote_call_all_workers_async,
            )
        else:
            self.block_manager = None
            logger.info("Enable unified block manager, Decodesince the MPS merged the GPU memory")
        
        self.scheduler: ContextStageScheduler | DecodingStageScheduler = self._get_scheduler()

        logger.info(f"Scheduler: {self.scheduler}")
        logger.info(f"Block manager: {self.block_manager}")

    def set_block_manager(self, block_manager):
        self.scheduler.set_block_manager(block_manager)
        self.block_manager = block_manager

    def get_block_manager(self):
        return self.block_manager
    
    
    async def restart_back_up_workers(self, new_percentage):
        """
        for each pipeline stage, create tensor_parallel_size workers
        each worker will be assigned a GPU
        the worker will be placed in the corresponding placement group
        """
        logger.info("Initializing workers")

        layer_per_placement_group = self.model_config.get_num_layers() // len(self.placement_groups)
        layer_per_pp = self.model_config.get_num_layers(self.parallel_config)
        pp_per_placement_group = layer_per_placement_group // layer_per_pp
        # create unique id
        pp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        
        init_handlers = []
        # workaround, since ray don't have ability to specific set gpu id, so we need to fulfill server
        # first, then kill the worker we need and restart
        for tp_workers in self.workers_back_up:
            for worker in tp_workers:
                worker.exit.remote()

                while True:
                    try:
                        ray.get(worker.get_state.remote())
                    except ray.exceptions.RayActorError:
                        print("Actor has been killed.")
                        break
                    except Exception as e:
                        print(f"Exception occurred: {e}")
                        break
                    time.sleep(0.1)

        self.workers_back_up = []
        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []
            placement_group_index = i // pp_per_placement_group
            tp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            cur_placement_group = self.placement_groups[placement_group_index]
            for j in range(self.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(self.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j
                worker = ParaWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=cur_placement_group,
                        placement_group_capture_child_tasks=True,
                    )
                ).remote()
                worker.init.remote(
                    worker_id=(i*self.parallel_config.tensor_parallel_size+j),
                    stage=self.stage,
                    model_config=self.model_config,
                    cache_config=self.cache_config,
                    parallel_config=tmp_parallel_config,
                    pipeline_parallel_id=pp_id,
                    tensor_parallel_id=tp_id,
                    peer_ids=self.peer_ids,
                    thread_percentile=new_percentage,
                )
                workers.append(worker)
                init_handlers.append(worker.ready.remote())
            self.workers_back_up.append(workers)
            
        await asyncio.wait(init_handlers)
    
    @classmethod
    async def collective_init_workers(cls, prefill_instance, decode_instance, prefill_thread_percentile, decode_thread_percentile):
        logger.info("Collective initialize workers")

        layer_per_placement_group = (prefill_instance.model_config.get_num_layers() + len(prefill_instance.placement_groups) - 1)// len(prefill_instance.placement_groups)
        layer_per_pp = prefill_instance.model_config.get_num_layers(prefill_instance.parallel_config)
        pp_per_placement_group = layer_per_placement_group // layer_per_pp
        # create unique id
        pp_id_prefill = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        pp_id_decode = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        pp_id_prefill_bak = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        pp_id_decode_bak = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        pp_id_prefill_bak_1 = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())

        init_handlers = []

        for i in range(prefill_instance.parallel_config.pipeline_parallel_size):
            prefill_workers = []
            decode_workers = []
            prefill_workers_back_up = []
            prefill_workers_back_up_1 = []
            decode_workers_back_up = []
            placement_group_index = i // pp_per_placement_group
            tp_id_prefill = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            tp_id_decode = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            tp_id_prefill_bak = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            tp_id_decode_bak = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            tp_id_prefill_bak_1 = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())

            cur_placement_group = prefill_instance.placement_groups[placement_group_index]

            workers_num_per_device =  5 if ENABLE_DYNAMIC_SWITCH else 2
            prefill_workers_num_per_device =  3 if ENABLE_DYNAMIC_SWITCH else 1
            decode_workers_num_per_device =  2 if ENABLE_DYNAMIC_SWITCH else 1
            
            initialized_workers_map = {}
            task_list = []
            nsight_runtime_env={"env_vars": {"ENABLE_MPS": str(ENABLE_MPS), 
                                      "ENABLE_DYNAMIC_SWITCH": str(ENABLE_DYNAMIC_SWITCH)},
                                "nsight": {
                                        "cuda-graph-trace": "node",
                                        "t": "cuda,cudnn,cublas,nvtx",
                                        "o": "'worker_process_%p'",
                                        # "cudabacktrace": "all",
                                        # "stop-on-exit": "true",
                                    }
                }
            default_runtime_env = {"env_vars": {"ENABLE_MPS": str(ENABLE_MPS), 
                                      "ENABLE_DYNAMIC_SWITCH": str(ENABLE_DYNAMIC_SWITCH)}}
            for j in range(prefill_instance.parallel_config.tensor_parallel_size):
                async def _create_worker():
                    tmp_parallel_config = copy.deepcopy(prefill_instance.parallel_config)
                    tmp_parallel_config.pipeline_parallel_rank = i
                    tmp_parallel_config.tensor_parallel_rank = j
                    # 2 prefill worker and 2 decode worker in one device
                    # this aims for dynamic cast mps SM partition percentage
                    for n in range(workers_num_per_device):
                        worker = ParaWorker.options(
                                scheduling_strategy=PlacementGroupSchedulingStrategy(
                                    placement_group=cur_placement_group,
                                    placement_group_capture_child_tasks=True,
                                ),
                                # runtime_env=nsight_runtime_env if n == 0 else default_runtime_env
                                runtime_env=default_runtime_env
                            ).remote()
                        gpu_id = ray.get(worker.get_device_id.remote())
                        item = initialized_workers_map.get(gpu_id, [])
                        item.append(worker)
                        initialized_workers_map[gpu_id] = item
                task_list.append(asyncio.create_task(_create_worker()))
            
            await asyncio.gather(*task_list)
            
            gpu_ids = list(initialized_workers_map.keys())
            
            for j in range(prefill_instance.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(prefill_instance.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j
                for k in range(prefill_workers_num_per_device):
                    prefill_worker = initialized_workers_map[gpu_ids[j]][k]
                    prefill_worker.init.remote(
                        worker_id=(i*prefill_instance.parallel_config.tensor_parallel_size+j),
                        stage=prefill_instance.stage,
                        model_config=prefill_instance.model_config,
                        cache_config=prefill_instance.cache_config,
                        parallel_config=tmp_parallel_config,
                        pipeline_parallel_id=pp_id_prefill if k ==0 else [pp_id_prefill_bak, pp_id_prefill_bak_1][k-1],
                        tensor_parallel_id=tp_id_prefill if k ==0 else [tp_id_prefill_bak, tp_id_prefill_bak_1][k-1],
                        peer_ids=prefill_instance.peer_ids,
                        thread_percentile=prefill_thread_percentile,
                    )
                    if k == 0:
                        prefill_workers.append(prefill_worker)
                    elif k == 1:
                        prefill_workers_back_up.append(prefill_worker)
                    else:
                        prefill_workers_back_up_1.append(prefill_worker)
                    init_handlers.append(prefill_worker.ready.remote())
                for k in range(decode_workers_num_per_device):
                    tmp_parallel_config = copy.deepcopy(decode_instance.parallel_config)
                    tmp_parallel_config.pipeline_parallel_rank = i
                    tmp_parallel_config.tensor_parallel_rank = j
                    decode_worker = initialized_workers_map[gpu_ids[j]][k + prefill_workers_num_per_device]
                    decode_worker.init.remote(
                        worker_id=(i*decode_instance.parallel_config.tensor_parallel_size+j),
                        stage=decode_instance.stage,
                        model_config=decode_instance.model_config,
                        cache_config=decode_instance.cache_config,
                        parallel_config=tmp_parallel_config,
                        pipeline_parallel_id=pp_id_decode if k == 0 else pp_id_decode_bak,
                        tensor_parallel_id=tp_id_decode if k == 0 else tp_id_decode_bak,
                        peer_ids=decode_instance.peer_ids,
                        thread_percentile=decode_thread_percentile,
                    )
                    
                    if k == 0:
                        decode_workers.append(decode_worker)
                    else:
                        decode_workers_back_up.append(decode_worker)

                    init_handlers.append(decode_worker.ready.remote())
            prefill_instance.workers.append(prefill_workers)
            decode_instance.workers.append(decode_workers)
            
            prefill_instance.workers_back_up.append(prefill_workers_back_up)
            decode_instance.workers_back_up.append(decode_workers_back_up)
            
            prefill_instance.workers_back_up_1.append(prefill_workers_back_up_1)


    async def _init_workers(self):
        """
        for each pipeline stage, create tensor_parallel_size workers
        each worker will be assigned a GPU
        the worker will be placed in the corresponding placement group
        """
        logger.info("Initializing workers")

        layer_per_placement_group = self.model_config.get_num_layers() // len(self.placement_groups)
        layer_per_pp = self.model_config.get_num_layers(self.parallel_config)
        pp_per_placement_group = layer_per_placement_group // layer_per_pp
        # create unique id
        pp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        
        init_handlers = []
        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []
            placement_group_index = i // pp_per_placement_group
            tp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            cur_placement_group = self.placement_groups[placement_group_index]
            for j in range(self.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(self.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j
                worker = ParaWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=cur_placement_group,
                        placement_group_capture_child_tasks=True,
                    )
                ).remote(
                    worker_id=(i*self.parallel_config.tensor_parallel_size+j),
                    stage=self.stage,
                    model_config=self.model_config,
                    cache_config=self.cache_config,
                    parallel_config=tmp_parallel_config,
                    pipeline_parallel_id=pp_id,
                    tensor_parallel_id=tp_id,
                    peer_ids=self.peer_ids,
                )
                workers.append(worker)
                init_handlers.append(worker.ready.remote())
            self.workers.append(workers)
            
        await asyncio.wait(init_handlers)

    async def _init_model(self, bypass_weigit_init=False):
        """
        init model by call init_model() on all workers
        """
        handlers = self._remote_call_all_workers_async("init_model", bypass_weigit_init)
        await asyncio.wait(handlers)

    async def _init_kvcache(self, bypass_block_init=False):
        """
        Profile available blocks and initialize k/v cache on all workers
        """
        logger.info("Profiling available blocks")
        num_gpu_blocks, num_cpu_blocks = await self.workers[0][0]._profile_num_available_blocks.remote(
            self.cache_config.block_size,
            self.cache_config.gpu_memory_utilization,
            self.cache_config.cpu_swap_space,
        )
            
        logger.info(f"Profiling result: num_gpu_blocks: {num_gpu_blocks}, num_cpu_blocks: {num_cpu_blocks}")
        # if self.stage == Stage.CONTEXT:
        #     # Do not set to 0 to avoid division by 0
        #     logger.info(f"The engine performs context stage, setting num_cpu_blocks to 1")
        # num_cpu_blocks = 1
        # num_cpu_blocks = 1024
        num_cpu_blocks = 4096
        logger.info("Allocating kv cache")
        kv_cache_mem_handles_1d = await asyncio.gather(*self._remote_call_all_workers_async(
            "init_kvcache_and_swap", num_gpu_blocks, num_cpu_blocks, bypass_block_init
        ))
        
        if not bypass_block_init:
            # Gather the address of kv cache for block migration
            self.kv_cache_mem_handles = []
            for stage in self.workers:
                kv_cache_mem_handles = []
                for worker in stage:
                    kv_cache_mem_handles.append(kv_cache_mem_handles_1d.pop(0))
                self.kv_cache_mem_handles.append(kv_cache_mem_handles)
        
        return num_gpu_blocks, num_cpu_blocks

    async def _init_flatten_weight_handles(self):
        flatten_weight_mem_handles_list = await asyncio.gather(*self._remote_call_all_workers_async(
            "get_weight_ipc_handle"
        ))
        self.flatten_weight_mem_handles = []
        for stage in self.workers:
            flatten_weight_mem_handles = []
            for worker in stage:
                flatten_weight_mem_handles.append(flatten_weight_mem_handles_list.pop(0))
            self.flatten_weight_mem_handles.append(flatten_weight_mem_handles)

    async def register_weight_mem_handles(
        self,
        context_parallel_config: ParallelConfig,
        weight_mem_handles: List[List[cudaMemoryIpcHandle]]
    ):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """
        self.flatten_weight_mem_handles = weight_mem_handles
        await asyncio.wait(self._remote_call_all_workers_async(
            "register_weight_mem_handles",
            context_parallel_config,
            weight_mem_handles
        ))
        
    async def lazy_init_weight(self, enable_ipc_mem, is_alloc):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """
        await asyncio.wait(self._remote_call_all_workers_async(
            "lazy_init_weight", enable_ipc_mem, is_alloc
        ))

    def _remote_call_all_workers_async(self, func_name: str, *args):
        """
        call func_name asynchronously on all workers, return the futures immediately
        """
        is_backup = WorkersManager.get()
        handlers = []
        workers = self.workers if is_backup == 0 else self.workers_back_up
        for stage in workers:
            for worker in stage:
                handlers.append(getattr(worker, func_name).remote(*args))
        return handlers

    def abort_request(self, request_id: int):
        """
        abort_request: Abort one request and free its resources
        """
        # Currently there may be some race conditions here,
        # so we just do nothing
        # TODO. Implement request abortion
        logger.warn(f"Request abortion is not implemented yet")
        return
        self.scheduler.abort_request(request_id)
        self._free_request_resources(request_id, bypass=False)
    
    @abstractmethod
    async def start_event_loop(self):
        raise NotImplementedError()
    
    @abstractmethod
    async def print_engine_status(self):
        raise NotImplementedError()
    
    async def register_kvcache_mem_handles(
        self,
        context_parallel_config: ParallelConfig,
        kv_cache_mem_handles: List[List[Tuple[cudaMemoryIpcHandle, cudaMemoryIpcHandle]]]
    ):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """
        self.kv_cache_mem_handles = kv_cache_mem_handles
        await asyncio.wait(self._remote_call_all_workers_async(
            "register_kvcache_mem_handles",
            context_parallel_config,
            kv_cache_mem_handles
        ))
    
    async def get_shared_kv_tensor(self):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """

        await asyncio.wait(self._remote_call_all_workers_async(
            "get_shared_kv_tensor",
        ))
    
    async def init_back_up_workers(self):
        with WorkersContext(1):
            await self._init_model(bypass_weigit_init=True)
            await self._init_kvcache(bypass_block_init=True)
            await self.register_kvcache_mem_handles(
                self.parallel_config,
                self.kv_cache_mem_handles
            )
            await self.register_weight_mem_handles(
                self.parallel_config,
                self.flatten_weight_mem_handles,
            )
            await self.lazy_init_weight(True, is_alloc=False)
            await self.get_shared_kv_tensor()
    
class ContextStageLLMEngine(SingleStageLLMEngine):
    def _get_scheduler(self) -> ContextStageScheduler:
        return get_context_stage_scheduler(
            self.sched_config,
            self.parallel_config,
            self.block_manager
        )
    
    def __init__(
        self,
        bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig,
        placement_groups: List[PlacementGroup],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None],
        peer_ids = None,
    ):
        super().__init__(
            Stage.CONTEXT,
            model_config,
            parallel_config,
            cache_config,
            sched_config,
            placement_groups,
            engine_on_new_step_output_callback,
            engine_on_new_lifetime_event_callback,
            peer_ids = peer_ids
        )
        
        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []
        
        self.bridge_queue = bridge_queue
        
        self._decode_semaphore_call_back_fn = None
        
        self.ttft_window = []
        self.wait_window = []
        
    def add_request(self, request: Request):
        self.scheduler.add_request(request)
    
    def set_decode_semaphore_call_back(self, fn):
        self._decode_semaphore_call_back_fn = fn
    
    def _free_request_resources(self, request_id: int, bypass):
        if not bypass:
            super()._free_request_resources(request_id)


        
    async def _step(self):
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        
        Note2. Pipeline parallel is not tested yet
        """
        # pick next batch from scheduler
        batched_requests = self.scheduler.get_next_batch_and_pop()
        # print(len(batched_requests))
        if len(batched_requests) == 0:
            # Two cases may cause len(batched_requests) == 0:
            # 1. No request in the waiting queue
            # 2. No enough free blocks (e.g. the decoding stage is too slow)
            self.batches_in_pipeline.append(batched_requests)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_CONTEXT_NO_REQUEST)
        else:
            logger.info(f"(context) Forwarding with lengths {[len(request.prompt_token_ids) for request in batched_requests.requests]}")
            # allocate blocks as needed
            # logger.info("----------before-prefill-forward--------")
            # self.block_manager.print_block_usage()
            # self.block_manager.allocate_blocks_batched(batched_requests)
            
            # Log down the lifetime event
            for request in batched_requests.requests:
                self.engine_on_new_lifetime_event_callback(
                    request.request_id,
                    LifetimeEvent(LifetimeEventType.ContextBegin)
                )
            # push the batch into pipeline
            batched_requests.start_one_iteration(time.time())
            tokens_per_batch = batched_requests.requests[0].prompt_token_ids.__len__()
            self.batches_in_pipeline.append(batched_requests)
            remote_calls = self._remote_call_all_workers_async(
                "step",
                batched_requests.get_request_ids(),
                batched_requests.get_input_tokens_batched(),
                batched_requests.get_first_token_indexes(),
                self.block_manager.get_partial_block_table(
                    batched_requests.get_request_ids()
                ),
            )
            
            pp_size = self.parallel_config.pipeline_parallel_size
            tp_size = self.parallel_config.tensor_parallel_size
            # only the leader of the last stage return valid output, i.e., generated tokens ids
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                # No request in the batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                generated_tokens_ids = await self.batches_ret_futures[0]
                
                # logger.info("----------after-prefill-forward--------")
                # self.block_manager.print_block_usage()
                
                    
                end_time = time.time()
                latency = (end_time - self.batches_in_pipeline[0].start_time) * 1e3
                d = self.model_config.hf_config.hidden_size
                l = self.model_config.hf_config.num_hidden_layers
                batch_size = len(self.batches_in_pipeline[0].requests)
                # ffn_h = self.model_config.hf_config.ffn_dim
                ffn_h = 4
                h_scale = ffn_h / d
                
                bld = 1 * l * d
                prefill_flops = 0
                total_tokens = self.batches_in_pipeline[0].get_num_input_tokens()
                # for req in self.batches_in_pipeline[0].requests:
                #     s = req.get_num_input_tokens()
                #     prefill_flops += ((4 * s ** 2) + (8 + 4 * h_scale) * s * d) * bld
                # pp = self.parallel_config.pipeline_parallel_size
                # tp = self.parallel_config.tensor_parallel_size
                # Prefill_TflopS = prefill_flops / (latency * 1e-3) / (self.parallel_config.tensor_parallel_size * self.parallel_config.pipeline_parallel_size) * 1e-12
                # P_mfu =  Prefill_TflopS / 312
                # logger.info("promt run : tokens %d, batch %s, latency %s ms,  p_mfu %f, prefill_flops %f, pp %s, tp %s", total_tokens, batch_size, latency, P_mfu, prefill_flops, pp, tp)
                
                generated_tokens = []
                for gen_token_id in generated_tokens_ids:
                    try:
                        token = self.tokenizer.decode(gen_token_id)
                    except Exception as e:
                        print(f"(context) Warning: Cannot decode token with id {gen_token_id}. Error: {e}")
                        token = ""
                    generated_tokens.append(token)

                finished_batch = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(
                    generated_tokens, generated_tokens_ids, end_time
                )
                # req_ttfts = []
                # req_waitings = []
                for req in self.batches_in_pipeline[0].requests:
                    ttft = (req.last_step_time - req.arrival_time) * 1e3
                    
                    id = req.request_id
                    # req_ttfts.append(ttft)
                    waiting_latency = ttft - latency
                    # self.ttft_window.append(waiting_latency)
                    # self.ttft_window.append((ttft, latency, waiting_latency))
                    self.dyn_partitioner.update_prefill_info((ttft, latency, waiting_latency))
                    logger.info("Req-%s tokens %d batch_size %d total_batched_tokens %d: latency %f, waiting %f, ttft %f ms, ", id, len(req.prompt_token_ids), batch_size,total_tokens, latency, waiting_latency, ttft)

                
                import numpy as np
                # p90_ttft = np.percentile(np.array(self.ttft_window), 90)
                # p90_waiting = np.percentile(np.array(self.wait_window), 90)
                # do_dynamic_switch = False
                # self.ttft_window = sorted(self.ttft_window, key = lambda x:x[0])
                # p90_idx = len(self.ttft_window) * 0.9 / 1 + 1
                # logger.info("p90 ttft:%f", p90_ttft)
                # req_ids = self.batches_in_pipeline[0].requests.get_request_ids()
                
                # logger.info("promt run : ttft %f ms , waiting_time %s ms, latency %s ms, p90_ttft %f ms, tokens %d, batch %s,  p_mfu %f, prefill_flops %f, pp %s, tp %s", ttft, waiting_time, latency, p90_ttft,  total_tokens, batch_size, P_mfu, prefill_flops, pp, tp)
                # logger.info("promt run : ttft %f ms , waiting_time %s ms, latency %s ms, p90_ttft %f ms, tokens %d, batch %s,  p_mfu %f, prefill_flops %f, pp %s, tp %s", ttft, waiting_time, latency, p90_ttft,  total_tokens, batch_size, P_mfu, prefill_flops, pp, tp)


                self.scheduler.on_finish_requests(finished_batch)
                
                for request, new_token, new_token_id in zip(
                    finished_batch.requests, generated_tokens, generated_tokens_ids
                ):
                    step_output = StepOutput(request, new_token, new_token_id)
                    self.engine_on_new_lifetime_event_callback(
                        request.request_id,
                        LifetimeEvent(LifetimeEventType.ContextEnd)
                    )
                    self.engine_on_new_step_output_callback(
                        request.request_id,
                        step_output
                    )

                # Cannot free blocks now! The decoding stage may still need them!

                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
                
                # Inform the user that the request has finished the context stage
                for request in finished_batch.requests:
                    if not request.is_finished:
                        # Push the request into the bridge queue if it is not finished
                        migrating_req = MigratingRequest(
                            request,
                            self.block_manager.get_block_table(request.request_id),
                            self.parallel_config,
                        )
                        self.bridge_queue.put_nowait(migrating_req) # This won't panic because the queue is unbounded
                    else:
                        self._free_request_resources(request.request_id, bypass=False)
                        
                do_dynamic_switch = False
                if ENABLE_DYNAMIC_SWITCH:
                    self.dyn_partitioner.update_prefill_status(self.switch_semaphore)
                    do_dynamic_switch = self.dyn_partitioner.check_switch_prefill()
                
                if self.switch_semaphore == SwitchSemaphore.REJECT:
                    if do_dynamic_switch:
                        # self.switch_percentile = 100
                        self.switch_percentile = self.dyn_partitioner.get_prefill_predicted_sm()
                        logger.info("\033[1;32;40m Prefill instance triggered switch %d->%d\033[0m",  self.sm_percentile,self.switch_percentile)
                        self.switch_semaphore = SwitchSemaphore.ACCEPT
                if self.switch_semaphore == SwitchSemaphore.END_INIT:
                    await self.worker_switch()
                    self.dyn_partitioner.prefill_cur_sm_percentile = self.switch_percentile
                    self.sm_percentile = self.switch_percentile
                    # reset
                    self.dyn_partitioner.do_prefill_replan = False
                    self.switch_semaphore = SwitchSemaphore.REJECT
    
    def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest, bypass):
        """
        Called when the decoding engine finishes migrating the blocks of the request.
        """
        self._free_request_resources(migrated_request.req.request_id, bypass)
        self.scheduler.on_request_migrated(migrated_request)

    
    async def _switch_worker_with_semaphore(self):
        if self.switch_semaphore == SwitchSemaphore.ACCEPT:
            self.switch_semaphore = SwitchSemaphore.START_INIT
            await self.restart_back_up_workers(self.switch_percentile)
            await self.init_back_up_workers()
            self.switch_semaphore = SwitchSemaphore.END_INIT
      
    async def start_event_loop(self):
        async def event_loop1():
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        
        async def event_loop2():
            while True:
                # self.print_engine_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)
        
        async def event_loop3():
            while True:
                await self._switch_worker_with_semaphore()
                # await asyncio.sleep(1)
                await asyncio.sleep(SLEEP_FOR_MPS_RESCHEDULE)

        await asyncio.gather(event_loop1(), event_loop2(), event_loop3())
        
    def print_engine_status(self):
        self.scheduler.print_status()
        
    async def register_kvcache_mem_handles(
        self,
        context_parallel_config: ParallelConfig,
        kv_cache_mem_handles: List[List[Tuple[cudaMemoryIpcHandle, cudaMemoryIpcHandle]]]
    ):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """
        self.kv_cache_mem_handles = kv_cache_mem_handles
        await asyncio.wait(self._remote_call_all_workers_async(
            "register_kvcache_mem_handles",
            context_parallel_config,
            kv_cache_mem_handles
        ))
    
    async def get_shared_kv_tensor(self):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """

        await asyncio.wait(self._remote_call_all_workers_async(
            "get_shared_kv_tensor",
        ))
        

class DecodingStageLLMEngine(SingleStageLLMEngine):
    
    def _get_scheduler(self) -> DecodingStageScheduler:
        return get_decoding_stage_scheduler(
            self.sched_config,
            self.parallel_config,
            self.block_manager,
            self._migrate_blocks
        )
        
    def __init__(
        self,
        bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: DecodingStageSchedConfig,
        placement_groups: List[PlacementGroup],
        clear_migrated_blocks_callback: Callable[[Request], None],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None],
        peer_ids = None,
    ):
        super().__init__(
            Stage.DECODING,
            model_config,
            parallel_config,
            cache_config,
            sched_config,
            placement_groups,
            engine_on_new_step_output_callback,
            engine_on_new_lifetime_event_callback,
            peer_ids=peer_ids
        )
        
        self.bridge_queue = bridge_queue
        self.clear_migrated_blocks_callback = clear_migrated_blocks_callback
        
        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline = []
        self.batches_ret_futures = []
        
        self._prefill_semaphore_call_back_fn = None
        
        self.topt_window = []
        # self.sm_percentile = 100
        self.gather_info = True
        
    def set_prefill_semaphore_call_back(self, fn):
        self._prefill_semaphore_call_back_fn = fn

    async def register_kvcache_mem_handles(
        self,
        context_parallel_config: ParallelConfig,
        kv_cache_mem_handles: List[List[Tuple[cudaMemoryIpcHandle, cudaMemoryIpcHandle]]]
    ):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """
        self.kv_cache_mem_handles = kv_cache_mem_handles
        await asyncio.wait(self._remote_call_all_workers_async(
            "register_kvcache_mem_handles",
            context_parallel_config,
            kv_cache_mem_handles
        ))
        
    async def get_shared_kv_tensor(self):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """

        await asyncio.wait(self._remote_call_all_workers_async(
            "get_shared_kv_tensor",
        ))
        
    async def get_shared_weight_tensor(self, tensor_size):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """
        
        await asyncio.wait(self._remote_call_all_workers_async(
            "get_shared_weight_tensor",tensor_size
        ))
    
    def _free_request_resources(self, request_id: int, bypass):
        # bypass = True
        # if not bypass:
        #     super()._free_request_resources(request_id)
        self.request_events.pop(request_id)
        self.request_outputs.pop(request_id)

    async def _migrate_blocks(
        self,
        migrating_req: MigratingRequest
    ) -> None:
        """
        Migrate one request from the context engine to the decoding engine
        
        This function will be called be the decoding stage scheduler
        
        This function performs the following steps:
        - Allocate blocks on the decoding engine's side
        - Transfer the blocks
        - Clear the blocks on the context engine's side
        """
        # Allocate blocks on the decoding engine's side
        
        # Here we temporarily backup the generated tokens and generated token ids
        # since we are going to overwrite them later when allocating blocks
        generated_token_bkup = migrating_req.req.generated_tokens
        generated_token_ids_bkup = migrating_req.req.generated_token_ids
        migrating_req.req.generated_tokens = []
        migrating_req.req.generated_token_ids = []
        # logger.info("----------before-migrate_blocks--------")
        # self.block_manager.print_block_usage()
        self.block_manager.allocate_blocks(migrating_req.req)
        migrating_req.req.generated_tokens = generated_token_bkup
        migrating_req.req.generated_token_ids = generated_token_ids_bkup
        
        target_block_indexes = self.block_manager.get_block_table(migrating_req.req.request_id)
        assert len(target_block_indexes) == len(migrating_req.block_indexes)
        # logger.info("target_block_indexes %s, \n migrate_block_indexes %s", target_block_indexes, migrating_req.block_indexes)
        # bypass = True
        # bypass = False
        if not self.bypass_block_init:
            # Transfer the blocks
            self.engine_on_new_lifetime_event_callback(
                migrating_req.req.request_id,
                LifetimeEvent(LifetimeEventType.MigrationBegin)
            )
            # logger.info("----------before-migrate_blocks--------")
            # self.block_manager.print_block_usage()
            await asyncio.wait(self._remote_call_all_workers_async(
                "migrate_blocks",
                migrating_req.block_indexes,
                migrating_req.context_parallel_config,
                target_block_indexes
            ))
            # logger.info("----------after-migrate_blocks--------")
            # self.block_manager.print_block_usage()
            self.engine_on_new_lifetime_event_callback(
                migrating_req.req.request_id,
                LifetimeEvent(LifetimeEventType.MigrationEnd)
            )
        
            # Clear the blocks on the context engine's side
            self.clear_migrated_blocks_callback(migrating_req, bypass=False)
        else:
            self.engine_on_new_lifetime_event_callback(
                migrating_req.req.request_id,
                LifetimeEvent(LifetimeEventType.MigrationEnd)
            )
            self.clear_migrated_blocks_callback(migrating_req, bypass=True)
            
    async def _step(self) -> None:
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """

        pp_size = self.parallel_config.pipeline_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size

        # pick next batch from scheduler
        # this may trigger migration if some requests are still at context stage
        # this may trigger swap_in if some requests have been swapped out to CPU
        # this may also trigger swap_out if GPU blocks are not enough
        batched_requests = self.scheduler.get_next_batch()

        if len(batched_requests) == 0:
            self.batches_in_pipeline.append(batched_requests)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_DECODING_NO_REQUEST)
        else:
            # Log down the lifetime event
            for request in batched_requests.requests:
                self.engine_on_new_lifetime_event_callback(
                    request.request_id,
                    LifetimeEvent(LifetimeEventType.DecodingBegin),
                    True
                )
                
            # Allocate blocks as needed
            # self.block_manager.allocate_blocks_batched(batched_requests)

            # Check if all requests are on GPU (i.e. not swapped out)
            assert self.block_manager.is_all_requests_on_gpu(
                batched_requests
            ), "Some requests are currently swapped out to CPU"

            # push the batch into pipeline
            batched_requests.start_one_iteration(time.time())
            self.batches_in_pipeline.append(batched_requests)
            # logger.info("----------before-decode-forward--------")
            # self.block_manager.print_block_usage()
            remote_calls = self._remote_call_all_workers_async(
                "step",
                batched_requests.get_request_ids(),
                batched_requests.get_input_tokens_batched(),
                batched_requests.get_first_token_indexes(),
                self.block_manager.get_partial_block_table(
                    batched_requests.get_request_ids()
                ),
            )
            # only the leader of the last stage return valid output, i.e., generated tokens ids
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        # output buffer
        finished_reqs = []

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                generated_tokens_ids = await self.batches_ret_futures[0]
                # logger.info("----------after-decode-forward--------")
                # self.block_manager.print_block_usage()
                end_time = time.time()
                latency = (end_time - self.batches_in_pipeline[0].start_time) * 1e3
                
                latency = (end_time - self.batches_in_pipeline[0].start_time) * 1e3
                d = self.model_config.hf_config.hidden_size
                l = self.model_config.hf_config.num_hidden_layers
                batch_size = len(self.batches_in_pipeline[0].requests)
                # ffn_h = self.model_config.hf_config.ffn_dim
                ffn_h = 4
                h_scale = ffn_h / d
                
                bld = 1 * l * d
                decode_flops = 0
                total_tokens = batch_size
                for req in self.batches_in_pipeline[0].requests:
                    s = req.get_kvcache_slots()
                    decode_flops += (4 * s + (8 + 4 * h_scale) * 1 * d) * bld

                pp = self.parallel_config.pipeline_parallel_size
                tp = self.parallel_config.tensor_parallel_size
                Decode_TflopS = decode_flops / (latency * 1e-3) / (self.parallel_config.tensor_parallel_size * self.parallel_config.pipeline_parallel_size) * 1e-12
                P_mfu =  Decode_TflopS / 312
                # logger.info("decode run : tokens %d, batch %s, latency %s ms,  p_mfu %f, decode_flops %f, pp %s , tp %s is_context %s", total_tokens, batch_size, latency, P_mfu, decode_flops, pp, tp, self.parallel_config.is_context)
                # logger.info("batch_size %d, latency %d ms", batch_size, latency)
                

                generated_tokens = []
                for gen_token_id in generated_tokens_ids:
                    try:
                        token = self.tokenizer.decode(gen_token_id)
                    except Exception as e:
                        print(f"(decoding) Warning: Cannot decode token with id {gen_token_id}. Error: {e}")
                        token = ""
                    generated_tokens.append(token)

                finished_batch : BatchedRequests = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(
                    generated_tokens, generated_tokens_ids, end_time
                )

                do_dynamic_switch = False
                
                self.dyn_partitioner.update_decode_status(self.switch_semaphore)
                
                for request, new_token, new_token_id in zip(
                    finished_batch.requests, generated_tokens, generated_tokens_ids
                ):
                    self.engine_on_new_step_output_callback(
                        request.request_id,
                        StepOutput(request, new_token, new_token_id)
                    )
                    if request.is_finished:
                        gen_len = len(request.generated_token_ids)
                        tpot = (request.process_time / gen_len) * 1e3
                        logger.info("Req-%s tpot %f ms, ", request.request_id, tpot)
                        if ENABLE_DYNAMIC_SWITCH:
                            gen_len = len(request.generated_token_ids)
                            tpot = (request.process_time / gen_len) * 1e3
                            self.dyn_partitioner.update_decode_info(tpot)

                        self.engine_on_new_lifetime_event_callback(
                            request.request_id,
                            LifetimeEvent(LifetimeEventType.DecodingEnd)
                        )
                
                finished_reqs = self.scheduler.pop_finished_requests()

                # free blocks for finished requests
                self.block_manager.free_blocks_batched(finished_reqs)
                self._remote_call_all_workers_async(
                    "clear_request_resource_batched", finished_reqs
                )
                
                self.dyn_partitioner.replan()
                do_dynamic_switch = self.dyn_partitioner.check_switch_decode()
                

                if self.switch_semaphore == SwitchSemaphore.REJECT:
                    if do_dynamic_switch:
                        self.switch_percentile = self.dyn_partitioner.get_decode_predicted_sm()
                        logger.info("\033[1;32;40m Decode instance triggered switch %d->%d\033[0m",  self.sm_percentile,self.switch_percentile)
                        self.switch_semaphore = SwitchSemaphore.ACCEPT
                if self.switch_semaphore == SwitchSemaphore.END_INIT:
                    await self.worker_switch()
                    self.dyn_partitioner.decode_cur_sm_percentile = self.switch_percentile
                    self.sm_percentile = self.switch_percentile
                    # reset
                    self.dyn_partitioner.do_decode_replan = False
                    self.switch_semaphore = SwitchSemaphore.REJECT
                
                self.step_count += 1
                # pop the finished batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)

        # proactive request migraion
        await self.scheduler.post_process()
        
    async def _switch_worker_with_semaphore(self):
        if self.switch_semaphore == SwitchSemaphore.ACCEPT:
            self.switch_semaphore = SwitchSemaphore.START_INIT
            await self.restart_back_up_workers(self.switch_percentile)
            await self.init_back_up_workers()
            self.switch_semaphore = SwitchSemaphore.END_INIT
    
    async def start_event_loop(self):
        async def event_loop1():
            # Event loop 1. Add migrating request to the scheduler
            while True:
                migrating_req = await self.bridge_queue.get()
                await self.scheduler.add_request(migrating_req)
                self.bridge_queue.task_done()
        
        async def event_loop2():
            # Event loop 2. Run step()
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        
        async def event_loop3():
            # Event loop 3. Print engine status
            while True:
                self.print_engine_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)
        
        
        async def event_loop4():
            while True:
                await self._switch_worker_with_semaphore()
                # await asyncio.sleep(1)
                await asyncio.sleep(SLEEP_FOR_MPS_RESCHEDULE)
        
        await asyncio.gather(event_loop1(), event_loop2(), event_loop3(), event_loop4())
    
    def print_engine_status(self):
        self.block_manager.print_block_usage()
        self.scheduler.print_status()
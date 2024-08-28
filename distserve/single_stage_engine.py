import time, copy
from typing import Callable, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import asyncio

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
from distserve.worker import ParaWorker
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
    
    async def initialize(self, bypass_block_init=False, bypass_worker_init=False):
        """Initialize workers, load models and initialize k/v cache
        
        We seperate this function from __init__ because we want to run it in an async way
        to enable parallel initialization between Engines.
        """
        if not bypass_worker_init:
            logger.info(f"Initializing {self.stage.name} workers")
            await self._init_workers()
        
        logger.info(f"Initializing {self.stage.name} models")
        await self._init_model()
        
        logger.info(f"Initializing {self.stage.name} kvcaches")
        self.num_gpu_blocks, self.num_cpu_blocks = await self._init_kvcache(bypass_block_init)
        # self.num_gpu_blocks, self.num_cpu_blocks = await self._init_kvcache(False)

        # if self.parallel_config.is_context == 1:
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
    
    @classmethod
    def collective_init_workers(cls, prefill_instance, decode_instance):
        logger.info("Collective initialize workers")

        layer_per_placement_group = prefill_instance.model_config.get_num_layers() // len(prefill_instance.placement_groups)
        layer_per_pp = prefill_instance.model_config.get_num_layers(prefill_instance.parallel_config)
        pp_per_placement_group = layer_per_placement_group // layer_per_pp
        # create unique id
        pp_id_prefill = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        pp_id_decode = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        init_handlers = []
        for i in range(prefill_instance.parallel_config.pipeline_parallel_size):
            prefill_workers = []
            decode_workers = []
            placement_group_index = i // pp_per_placement_group
            tp_id_prefill = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            tp_id_decode = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            cur_placement_group = prefill_instance.placement_groups[placement_group_index]
            for j in range(prefill_instance.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(prefill_instance.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j
                prefill_worker = ParaWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=cur_placement_group,
                        placement_group_capture_child_tasks=True,
                    )
                ).remote(
                    worker_id=(i*prefill_instance.parallel_config.tensor_parallel_size+j),
                    stage=prefill_instance.stage,
                    model_config=prefill_instance.model_config,
                    cache_config=prefill_instance.cache_config,
                    parallel_config=tmp_parallel_config,
                    pipeline_parallel_id=pp_id_prefill,
                    tensor_parallel_id=tp_id_prefill,
                    peer_ids=prefill_instance.peer_ids,
                )
                prefill_workers.append(prefill_worker)
                init_handlers.append(prefill_worker.ready.remote())
                
                tmp_parallel_config = copy.deepcopy(decode_instance.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j
                decode_worker = ParaWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=cur_placement_group,
                        placement_group_capture_child_tasks=True,
                    )
                ).remote(
                    worker_id=(i*decode_instance.parallel_config.tensor_parallel_size+j),
                    stage=decode_instance.stage,
                    model_config=decode_instance.model_config,
                    cache_config=decode_instance.cache_config,
                    parallel_config=tmp_parallel_config,
                    pipeline_parallel_id=pp_id_decode,
                    tensor_parallel_id=tp_id_decode,
                    peer_ids=decode_instance.peer_ids,
                )
                
                decode_workers.append(decode_worker)
                init_handlers.append(decode_worker.ready.remote())
            prefill_instance.workers.append(prefill_workers)
            decode_instance.workers.append(decode_workers)
            
        # asyncio.wait(init_handlers)

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

    async def _init_model(self):
        """
        init model by call init_model() on all workers
        """
        handlers = self._remote_call_all_workers_async("init_model")
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
        if self.stage == Stage.CONTEXT:
            # Do not set to 0 to avoid division by 0
            logger.info(f"The engine performs context stage, setting num_cpu_blocks to 1")
            num_cpu_blocks = 1
        logger.info("Allocating kv cache")
        kv_cache_mem_handles_1d = await asyncio.gather(*self._remote_call_all_workers_async(
            "init_kvcache_and_swap", num_gpu_blocks, num_cpu_blocks, bypass_block_init
        ))
        
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
        self.weight_mem_handles = weight_mem_handles
        await asyncio.wait(self._remote_call_all_workers_async(
            "register_weight_mem_handles",
            context_parallel_config,
            weight_mem_handles
        ))
        
    async def lazy_init_weight(self, enable_ipc_mem):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """
        await asyncio.wait(self._remote_call_all_workers_async(
            "lazy_init_weight", enable_ipc_mem
        ))

    def _remote_call_all_workers_async(self, func_name: str, *args):
        """
        call func_name asynchronously on all workers, return the futures immediately
        """
        handlers = []
        for stage in self.workers:
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
    
    def add_request(self, request: Request):
        self.scheduler.add_request(request)
    
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
            logger.info("----------before-prefill-forward--------")
            self.block_manager.print_block_usage()
            self.block_manager.allocate_blocks_batched(batched_requests)
            
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
                
                logger.info("----------after-prefill-forward--------")
                self.block_manager.print_block_usage()
                
                    
                end_time = time.time()
                latency = (end_time - batched_requests.start_time) * 1e3
                d = self.model_config.hf_config.hidden_size
                l = self.model_config.hf_config.num_hidden_layers
                batch_size = len(batched_requests.requests)
                ffn_h = self.model_config.hf_config.ffn_dim
                h_scale = ffn_h / d
                
                bld = 1 * l * d
                prefill_flops = 0
                total_tokens = batched_requests.get_num_input_tokens()
                for req in batched_requests.requests:
                    s = req.get_num_input_tokens()
                    prefill_flops += ((4 * s ** 2) + (8 + 4 * h_scale) * s * d) * bld
                pp = self.parallel_config.pipeline_parallel_size
                tp = self.parallel_config.tensor_parallel_size
                Prefill_TflopS = prefill_flops / (latency * 1e-3) / (self.parallel_config.tensor_parallel_size * self.parallel_config.pipeline_parallel_size) * 1e-12
                P_mfu =  Prefill_TflopS / 312
                logger.info("promt run : tokens %d, batch %s, latency %s ms,  p_mfu %f, prefill_flops %f, pp %s, tp %s", total_tokens, batch_size, latency, P_mfu, prefill_flops, pp, tp)
       
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
    
    def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest, bypass):
        """
        Called when the decoding engine finishes migrating the blocks of the request.
        """
        self._free_request_resources(migrated_request.req.request_id, bypass)
        self.scheduler.on_request_migrated(migrated_request)
        
    async def start_event_loop(self):
        async def event_loop1():
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        
        async def event_loop2():
            while True:
                # self.print_engine_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)

        await asyncio.gather(event_loop1(), event_loop2())
        
    def print_engine_status(self):
        self.scheduler.print_status()
        

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
        logger.info("----------before-migrate_blocks--------")
        self.block_manager.print_block_usage()
        self.block_manager.allocate_blocks(migrating_req.req)
        migrating_req.req.generated_tokens = generated_token_bkup
        migrating_req.req.generated_token_ids = generated_token_ids_bkup
        
        target_block_indexes = self.block_manager.get_block_table(migrating_req.req.request_id)
        assert len(target_block_indexes) == len(migrating_req.block_indexes)
        logger.info("target_block_indexes %s, \n migrate_block_indexes %s", target_block_indexes, migrating_req.block_indexes)
        bypass = True
        # bypass = False
        if not bypass:
            # Transfer the blocks
            self.engine_on_new_lifetime_event_callback(
                migrating_req.req.request_id,
                LifetimeEvent(LifetimeEventType.MigrationBegin)
            )
            logger.info("----------before-migrate_blocks--------")
            self.block_manager.print_block_usage()
            await asyncio.wait(self._remote_call_all_workers_async(
                "migrate_blocks",
                migrating_req.block_indexes,
                migrating_req.context_parallel_config,
                target_block_indexes
            ))
            logger.info("----------after-migrate_blocks--------")
            self.block_manager.print_block_usage()
            self.engine_on_new_lifetime_event_callback(
                migrating_req.req.request_id,
                LifetimeEvent(LifetimeEventType.MigrationEnd)
            )
        
            # Clear the blocks on the context engine's side
            self.clear_migrated_blocks_callback(migrating_req, bypass)
        else:
            self.engine_on_new_lifetime_event_callback(
                migrating_req.req.request_id,
                LifetimeEvent(LifetimeEventType.MigrationEnd)
            )
            self.clear_migrated_blocks_callback(migrating_req, bypass)
            
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
            self.block_manager.allocate_blocks_batched(batched_requests)

            # Check if all requests are on GPU (i.e. not swapped out)
            assert self.block_manager.is_all_requests_on_gpu(
                batched_requests
            ), "Some requests are currently swapped out to CPU"

            # push the batch into pipeline
            batched_requests.start_one_iteration(time.time())
            self.batches_in_pipeline.append(batched_requests)
            logger.info("----------before-decode-forward--------")
            self.block_manager.print_block_usage()
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
                logger.info("----------after-decode-forward--------")
                self.block_manager.print_block_usage()
                end_time = time.time()
                latency = (end_time - batched_requests.start_time) * 1e3
                
                latency = (end_time - batched_requests.start_time) * 1e3
                d = self.model_config.hf_config.hidden_size
                l = self.model_config.hf_config.num_hidden_layers
                batch_size = len(batched_requests.requests)
                ffn_h = self.model_config.hf_config.ffn_dim
                h_scale = ffn_h / d
                
                bld = 1 * l * d
                decode_flops = 0
                total_tokens = batch_size
                for req in batched_requests.requests:
                    s = req.get_kvcache_slots()
                    decode_flops += (4 * s + (8 + 4 * h_scale) * 1 * d) * bld

                pp = self.parallel_config.pipeline_parallel_size
                tp = self.parallel_config.tensor_parallel_size
                Decode_TflopS = decode_flops / (latency * 1e-3) / (self.parallel_config.tensor_parallel_size * self.parallel_config.pipeline_parallel_size) * 1e-12
                P_mfu =  Decode_TflopS / 312
                logger.info("decode run : tokens %d, batch %s, latency %s ms,  p_mfu %f, decode_flops %f, pp %s , tp %s is_context %s", total_tokens, batch_size, latency, P_mfu, decode_flops, pp, tp, self.parallel_config.is_context)
                generated_tokens = []
                for gen_token_id in generated_tokens_ids:
                    try:
                        token = self.tokenizer.decode(gen_token_id)
                    except Exception as e:
                        print(f"(decoding) Warning: Cannot decode token with id {gen_token_id}. Error: {e}")
                        token = ""
                    generated_tokens.append(token)

                finished_batch = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(
                    generated_tokens, generated_tokens_ids, end_time
                )

                for request, new_token, new_token_id in zip(
                    finished_batch.requests, generated_tokens, generated_tokens_ids
                ):
                    self.engine_on_new_step_output_callback(
                        request.request_id,
                        StepOutput(request, new_token, new_token_id)
                    )
                    if request.is_finished:
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

                # pop the finished batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)

        # proactive request migraion
        await self.scheduler.post_process()
    
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
                
        await asyncio.gather(event_loop1(), event_loop2(), event_loop3())
    
    def print_engine_status(self):
        self.block_manager.print_block_usage()
        self.scheduler.print_status()
        
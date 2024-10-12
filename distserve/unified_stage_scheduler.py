from distserve.request import Request, BatchedRequests, MigratingRequest
from distserve.logger import init_logger
logger = init_logger(__name__)


def _check_add_to_cur_batch(self, next_batch, request: Request, is_context, trigger_swap:bool = False, alloc_running :bool = False) -> bool:
    if not is_context:
        if alloc_running:
            check_and_alloc_current_runing(self)
            return None
        # condition 0:
        cond_0 = len(self.batch_queues[self.cur_index]) < self.sched_config.max_batch_size
        # condition 1:
        cond_1 = self.batch_queues[self.cur_index].get_num_input_tokens() + request.get_num_input_tokens() <= self.sched_config.max_tokens_per_batch
        # condition 2:
        cond_2 = (
            sum([
                sum([
                    self._get_block_needed(len(req.prompt_token_ids) + req.get_output_len())
                    for req in self.batch_queues[index].requests
                ])
                for index in range(self.parallel_config.pipeline_parallel_size)
            ]) + sum([
                self._get_block_needed(len(req.prompt_token_ids))
                for req in self.waiting_queue
            ]) + self._get_block_needed(request.get_input_len() + request.get_output_len()) \
                <= self.block_manager.max_num_gpu_blocks
        )
        # history_blocks = len(self.block_manager.block_table[request.request_id])  if request.request_id in self.block_manager.block_table else 0
        # print(f"request-{request.request_id} need: {self.block_manager.get_num_blocks_needed(request)}, actual: {self.block_manager.get_num_avail_gpu_blocks()}" )
        cond_3 = self.block_manager.get_num_blocks_needed(request) < self.block_manager.get_num_avail_gpu_blocks()
        can_add =  all([cond_0, cond_1, cond_2, cond_3])
        if can_add:
            if trigger_swap:
                logger.info("Swap-in triggered")
                # print(f"request-{request.request_id} need: {self.block_manager.get_num_blocks_needed(request)}, actual: {self.block_manager.get_num_avail_gpu_blocks()}" )
                self.block_manager.swap_in_requests([request])
                self.batch_queues[self.cur_index].add_request(request)
                self.block_manager.allocate_blocks(self.batch_queues[self.cur_index].requests[-1])
            else:
                self.batch_queues[self.cur_index].add_request(request)
                self.block_manager.allocate_blocks(self.batch_queues[self.cur_index].requests[-1])
        return can_add
    else:
        # logger.info("Context invoke _check_add_to_cur_batch")
        avail_blocks  =  self.block_manager.get_num_avail_gpu_blocks()
        cond_0 =  len(next_batch) < self.sched_config.max_batch_size
        cond_1 = (
            # Limit 2. tokens per batch
            next_batch.get_num_input_tokens()
            + request.get_num_input_tokens()
            <= self.sched_config.max_tokens_per_batch
        )
        cond_2 = (
                # # Limit 3. GPU blocks
                # sum([
                #     self._get_block_needed(len(req.prompt_token_ids))
                #     for req in next_batch.requests + [request]
                # ]) +
                # sum([
                #     self._get_block_needed(len(req.prompt_token_ids))
                #     for req in self.unaccepted_queue
                # ]) +
                # self.num_on_fly_request_block 
                # <= self.block_manager.max_num_gpu_blocks * 0.9
                sum([
                    self._get_block_needed(len(req.prompt_token_ids))
                    for req in next_batch.requests + [request]
                ])<= avail_blocks 
                # and avail_blocks > self.block_manager.max_num_gpu_blocks * 0.05
            )
    
        can_add =  all([cond_0, cond_1, cond_2])
        if can_add:
            self.block_manager.allocate_blocks(request)
            
        return can_add

def get_next_batch_and_pop(self) -> BatchedRequests:
    """
    Get the next batch for the context stage in a FCFS-like manner, and pop them
    """
    next_batch = BatchedRequests()

    while len(self.waiting_queue) > 0:
        request = self.waiting_queue[0]
        if self._check_add_to_cur_batch(next_batch, request, True):
            next_batch.add_request(request)
            self.waiting_queue.pop(0)
        else:
            break

    self.num_on_fly_request_block += sum([
        self._get_block_needed(req.get_input_len())
        for req in next_batch.requests
    ])

    return next_batch

def check_and_alloc_current_runing(self):
    self.cur_index = (
        self.cur_index + 1
    ) % self.parallel_config.pipeline_parallel_size

    # Check whether the blocks on GPU is enough for the next batch.
    # If not, swap out the last request
    # while sum([
    #     sum([
    #         self._get_block_needed(req.get_input_len() + req.get_output_len())
    #         for req in self.batch_queues[index].requests
    #     ])
    #     for index in range(self.parallel_config.pipeline_parallel_size)
    # ]) + sum([
    #     self._get_block_needed(req.get_input_len())
    #     for req in self.waiting_queue
    # ]) > self.block_manager.max_num_gpu_blocks:
    while sum([
        sum([
            self.block_manager.get_num_blocks_needed(req) - len(self.block_manager.block_table[req.request_id]) 
            for req in self.batch_queues[index].requests
        ])
        for index in range(self.parallel_config.pipeline_parallel_size)
    ]) > self.block_manager.get_num_avail_gpu_blocks():
        logger.info("No enough GPU blocks. Swap-out triggered")
        request = self.batch_queues[self.cur_index].requests.pop(-1)
        self.swapped_queue.append(request)
        self.block_manager.swap_out_requests([request])
    
    if len(self.batch_queues[self.cur_index]) != 0:
        self.block_manager.allocate_blocks_batched(self.batch_queues[self.cur_index])

def get_next_batch(self) -> BatchedRequests:
    # check_and_alloc_current_runing(self)
    self._check_add_to_cur_batch(None, request=None, is_context=False, trigger_swap=False, alloc_running=True)

    # Try to add in some new requests. Consider requests in the swapped queue first.
    while len(self.swapped_queue) > 0 or len(self.waiting_queue) > 0:
        if len(self.swapped_queue) > 0:
            request = self.swapped_queue[0]
            # check and alloc
            if self._check_add_to_cur_batch(None, request,is_context=False, trigger_swap=True):
                logger.info("Swap-in triggered")
                self.swapped_queue.pop(0)
            else:
                break
        else:
            request = self.waiting_queue[0]
            # check and alloc
            if self._check_add_to_cur_batch(None, request, is_context=False, trigger_swap=False):
                self.waiting_queue.pop(0)
            else:
                break
    return self.batch_queues[self.cur_index]
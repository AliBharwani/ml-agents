"""

        if self.trainer_settings.multiprocess_trainer:
            # This trainer process will just be a consumer of gpu trajectories
            # CREATE PRODUCER PROCESS
            self.gpu_batch_queue = simple_queue_with_size.SimpleQueueWithSize()
            self.dummy_q = torch.multiprocessing.SimpleQueue()
            # self.num_batches_in_q = mp.Value("i", 0)
            self.num_steps_processed = mp.Value("i", 0)
            batch_size_data = (self.wm_window, self.wm_batch_size, self.policy_window, self.policy_batch_size)
            gpu_batch_producer = mp.Process(
                target=SuperTrackTrainer.gpu_batch_producer_process,
                  name="gpu_batch_producer", 
                  args=(torch_settings, 
                        self.trajectory_queues[0], 
                        self.gpu_batch_queue,
                        # self.num_batches_in_q,
                        self._stats_reporter, 
                        self.num_steps_processed, 
                        self.hyperparameters.buffer_size,
                        batch_size_data, self.dummy_q), 
                  daemon=True)
            gpu_batch_producer.start()


    @staticmethod
    def gpu_batch_producer_process(torch_settings, traj_queue, gpu_batch_queue, stats_reporter, num_steps_processed, buffer_size, batch_size_data, dummy_q):
        logging_util.set_log_level(logging_util.INFO)
        torch_utils.set_torch_config(torch_settings)
        logger.info(f"gpu_batch_producer_process started on pid {os.getpid()} parent pid {os.getppid()}")
        gpu_buffer: AgentBuffer = AgentBuffer()
        wm_window, wm_batch_size, policy_window, policy_batch_size = batch_size_data   
        effective_wm_window = wm_window + 1
        effective_policy_window = policy_window + 1
        max_data_required = max(effective_wm_window * wm_batch_size, effective_policy_window * policy_batch_size)
        MAX_NUM_BATCHES_IN_Q = 4
        dummy_buffer = []
        try:
            while True:
                # read from traj_queue
                _queried = False
                num_read = 0
                num_steps_processed_this_iteration = 0
                processed_large_number_of_trajectories = traj_queue.qsize() > 150
                if (processed_large_number_of_trajectories):
                    print(f"{datetime.now().strftime('%I:%M:%S ')} Large number of trajectories in queue: {traj_queue.qsize()}")
                for _ in range(traj_queue.qsize()):
                    _queried = True
                    try:
                        traj = traj_queue.get_nowait()
                        # self._process_trajectory(t)
                        agent_buffer_trajectory = traj.to_supertrack_agentbuffer()
                        for st_datum in agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA]:
                            # print("Moving supertrack data on trajectory to GPU")
                            st_datum.to(default_device())
                        # Allocate a CUDA tensor to see if it fucks up memory
                        dummy_tensor = torch.ones( 30,  device=default_device())
                        dummy_buffer.append(dummy_tensor)
                        num_steps_processed_this_iteration += len(traj.steps)
                        agent_buffer_trajectory.resequence_and_append(
                           gpu_buffer, training_length=1
                        )
                        num_read += 1
                    except AgentManagerQueue.Empty:
                        break
                if not _queried:
                    # Yield thread to avoid busy-waiting
                    time.sleep(0.001)
                if (processed_large_number_of_trajectories):
                    print(f"{datetime.now().strftime('%I:%M:%S ')} Finished processing trajectories in queue, num_read: {num_read}")
                if num_read > 0:
                    stats_reporter.add_stat('Avg # Traj Read', num_read, StatsAggregationMethod.AVERAGE)

                # create batch and add to gpu_batch_queue
                if (gpu_buffer.num_experiences - max(effective_wm_window, effective_policy_window) >= max_data_required) and gpu_batch_queue.qsize() < MAX_NUM_BATCHES_IN_Q:
                    wm_minibatch = gpu_buffer.supertrack_sample_mini_batch(wm_batch_size, wm_window)
                    policy_minibatch = gpu_buffer.supertrack_sample_mini_batch(policy_batch_size, policy_window)
                    num_issue = 0
                    num_zero = 0
                    num_nan = 0
                    st_data = [wm_minibatch[BufferKey.SUPERTRACK_DATA][i] for i in range(wm_minibatch.num_experiences)]
                    for st_datum in st_data:
                        tensor = st_datum.sim_char_state.positions
                        num_zero += 1 if torch.count_nonzero(tensor).item() == 0 else 0
                        num_nan += 1 if torch.isnan(tensor).any() else 0
                        num_issue += 1 if ModelUtils.check_values_near_zero_or_nan(st_datum.sim_char_state.positions) else 0
                    # print(f"=========== PRODUCER THREAD: World model batch has {num_issue} issues out of {len(st_data)} possible") 
                    # print(f"=========== PRODUCER THREAD: World model batch has {num_zero} zero and {num_nan} nan out of {len(st_data)} possible") 
                    gpu_batch_queue.put((wm_minibatch, policy_minibatch))
                    dummy_batch = torch.stack(dummy_buffer[:1024]) # shape: [1024, 1, 30]
                    print(f"Producer dummy batch [:10] : {dummy_batch[:10]}")
                    # print(f"Dummy batch dtype: {dummy_batch.dtype}, device: {dummy_batch.device}")
                    idxes_of_zeroes = torch.argwhere(torch.where(dummy_batch.flatten(0, -1) == 0, 1, 0.)) 
                    # print(f"idxes_of_zeroes dtype: {idxes_of_zeroes.dtype}, device: {idxes_of_zeroes.device}")
                    print(f"=========== PRODUCER Idxes of zeroes [0]: {idxes_of_zeroes[0]} at [1]: {idxes_of_zeroes[-1]} median: {torch.median(idxes_of_zeroes)} mean: {torch.mean(idxes_of_zeroes.to(torch.float32))}")
                    print(f"=========== PRODUCER THREAD: World model batch has { torch.sum(dummy_batch == 0).item()} zero out of {dummy_batch.numel()} possible")
                    dummy_q.put(dummy_batch)
                    
                # Do this after setting up batch queue 
                num_steps_processed.value += num_steps_processed_this_iteration

                 # Truncate update buffer if neccessary. Truncate more than we need to to avoid truncating
                # a large buffer at each update.
                if gpu_buffer.num_experiences > buffer_size:
                    with hierarchical_timer("update_buffer.truncate"):
                        gpu_buffer.truncate_on_traj_end(
                            int(buffer_size * BUFFER_TRUNCATE_PERCENT)
                        )
                        logger.info(f"Truncated update buffer to {gpu_buffer.num_experiences} experiences")

        except KeyboardInterrupt:
            logger.info("gpu_batch_producer_process received KeyboardInterrupt")
        # finally:
            # prof.stop()
            # write_timing_tree(trainer.run_log_path)

    def advance_consumer(self):
        # Advances the consumer/training part of this trainer
        # return: whether or not the trainer read a batch 
        # this would normally happen in super._process_trajectory() but we do it manually here: 
        num_new_steps_processed = self.num_steps_processed.value
        if num_new_steps_processed > 0:
            self._maybe_write_summary(self.get_step + num_new_steps_processed)
            self._maybe_save_model(self.get_step + num_new_steps_processed)
            self._increment_step(num_new_steps_processed, self.policy_queues[0].behavior_id)
            self.num_steps_processed.value = 0
        _update_occured = False
        if self.should_still_train:
            if self._is_ready_update():
                with hierarchical_timer("_update_policy"):
                    # print(f"{datetime.now().strftime('%I:%M:%S ')} Entering trainer update policy")
                    batches = None
                    try:
                        batches = self.gpu_batch_queue.get()
                        dummy_tensor = self.dummy_q.get()
                        # print(f"======= TRAINER THREAD GOT DUMMY BATCH WITH NUM ZERO: {torch.sum(dummy_tensor == 0).item()} out of {dummy_tensor.numel()} possible")
                        idxes_of_zeroes = torch.argwhere(torch.where(dummy_tensor.flatten(0, -1) == 0, 1, 0.))
                        # print(f"======= TRAINER THREAD Idxes of zeroes [0]: {idxes_of_zeroes[0]} at [1]: {idxes_of_zeroes[-1]} median: {torch.median(idxes_of_zeroes)} mean: {torch.mean(idxes_of_zeroes.to(torch.float32))}")
                        # print(f"======= TRAINER THREAD GOT DUMMY BATCH WITH NUM ZERO: {torch.sum(dummy_tensor == 0).item()} out of {dummy_tensor.numel()} possible")
                        print(f"TRAINER dummy batch [:10] : {dummy_tensor[:10]}")

                        # print(dummy_tensor)
                        # Check if dummy_tensor is all ones
                        # if not torch.all(torch.eq(dummy_tensor, torch.ones_like(dummy_tensor))):
                            # print("ERROR: GPU batch producer process did not allocate a new tensor")
                            # print(dummy_tensor)
                            # raise Exception("GPU batch producer process did not allocate a new tensor")
                        # self.num_batches_in_q.value -= 1
                    except mp_queue.Empty:
                        pass
                    if batches is not None and self._update_policy(batches=batches, max_update_iterations=1): # only update once since provided one batch
                        del batches
                        _update_occured = True
                        # if self.profiler_state == ProfilerState.RUNNING: torch.cuda.nvtx.range_push("put in policy queue")
                        for q in self.policy_queues:
                            # Get policies that correspond to the policy queue in question
                            q.put(self.get_policy(q.behavior_id))
                        # if self.profiler_state == ProfilerState.RUNNING: torch.cuda.nvtx.range_pop()
                    # print(f"{datetime.now().strftime('%I:%M:%S ')} Exiting trainer update policy")
        return _update_occured
"""
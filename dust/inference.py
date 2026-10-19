"""Optional CUDA Graph inference with the original per-batch FPS randomness."""

from time import perf_counter

import torch

from models.fps_control import supplied_fps_starts


class InferenceForward:
    """Reuse one full-batch graph across frames; run ragged batches eagerly.

    The graph reads four FPS start tensors. Before every replay, draw those
    starts on CPU in the same order and shapes as the unchanged eager model.
    Warmup/capture consume no random numbers. Training never uses this class.
    """

    def __init__(self, model, device, batch_size, backend="eager"):
        if backend not in ("eager", "cuda-graph") or batch_size < 1:
            raise ValueError("Invalid inference backend/batch size")
        if backend == "cuda-graph" and device.type != "cuda":
            raise ValueError("cuda-graph evaluation requires a CUDA device; use eager on CPU")
        self.model, self.device = model, device
        self.batch_size, self.backend = batch_size, backend
        self.graph = self.static_input = self.static_output = None
        self.starts = None
        self.eager_batches = self.replays = 0
        self.capture_seconds = 0.

    def _capture(self, tensor):
        print("Capturing validation CUDA Graph (first full batch only)...", flush=True)
        clock = perf_counter()
        # PointNet++ MSG has four abstraction layers. Validate this explicit
        # contract instead of silently fixing the wrong model's randomness.
        try:
            counts = [tensor.shape[2]] + [getattr(self.model, f"sa{i}").npoint for i in (1, 2, 3)]
            getattr(self.model, "sa4").npoint
        except AttributeError as error:
            raise ValueError("CUDA Graph backend requires the four-layer MSG model") from error
        self.static_input = torch.empty_like(tensor)
        self.static_input.copy_(tensor)
        self.starts = [(count, torch.zeros(tensor.shape[0], dtype=torch.long, device=self.device))
                       for count in counts]
        current = torch.cuda.current_stream(self.device)
        warmup = torch.cuda.Stream(device=self.device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup):
            for _ in range(2):
                with supplied_fps_starts(self.starts):
                    self.model(self.static_input)
        current.wait_stream(warmup)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with supplied_fps_starts(self.starts):
                self.static_output = self.model(self.static_input)
        current.synchronize()
        self.capture_seconds = perf_counter() - clock
        print(f"Validation CUDA Graph ready in {self.capture_seconds:.1f}s", flush=True)

    @torch.inference_mode()
    def __call__(self, tensor):
        if self.model.training:
            raise RuntimeError("Inference backend requires model.eval()")
        if self.backend == "eager" or tensor.shape[0] != self.batch_size:
            self.eager_batches += 1
            return self.model(tensor)
        if self.graph is None:
            self._capture(tensor)
        if (tensor.shape != self.static_input.shape or tensor.dtype != self.static_input.dtype
                or tensor.device != self.static_input.device):
            raise ValueError("Graph inputs changed shape/dtype/device")
        self.static_input.copy_(tensor)
        for count, start in self.starts:
            # Retain one CPU randint call per layer, including its RNG state.
            start.copy_(torch.randint(0, count, (tensor.shape[0],), dtype=torch.long))
        self.graph.replay()
        # Output buffers belong to the graph and are overwritten on replay.
        # predict_frame copies scores to CPU before requesting the next batch.
        self.replays += 1
        return self.static_output

    def statistics(self):
        return {"backend": self.backend, "eager_batches": self.eager_batches,
                "graph_replays": self.replays, "graph_capture_seconds": self.capture_seconds}

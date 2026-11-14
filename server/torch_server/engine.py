"""The engine's C API (include/torch_engine.h) through ctypes: one handle, one request at a time, from one thread.

Every struct carries its size, so a library built from another version of the header refuses the call. Callbacks run
on the thread that called generate(); an exception inside one is kept and stops the request (ctypes would print and
swallow it, and the generation would go on).
"""
import ctypes
from ctypes import CFUNCTYPE, POINTER, Structure, c_char_p, c_double, c_float, c_int, c_int32, c_uint32, c_uint64
from ctypes import c_void_p

ABI_VERSION = 1
OK, ERROR, INVALID, CANCELLED, BUSY = range(5)
FINISH_LENGTH, FINISH_STOP, FINISH_CANCELLED = range(3)


class Options(Structure):
    _fields_ = [("struct_size", c_uint32), ("model", c_char_p), ("draft", c_char_p), ("draft_vocab", c_char_p),
                ("draft_head_rows", c_int32), ("placement", c_char_p), ("context", c_int32),
                ("prefill_columns", c_int32), ("lend_kv", c_int32), ("swaps_per_window", c_int32)]


class Params(Structure):
    _fields_ = [("struct_size", c_uint32), ("max_tokens", c_int32), ("speculative", c_int32), ("sample", c_int32),
                ("temperature", c_float), ("top_k", c_int32), ("top_p", c_float), ("min_p", c_float),
                ("repetition_penalty", c_float), ("presence_penalty", c_float), ("frequency_penalty", c_float),
                ("has_seed", c_int32), ("seed", c_uint64), ("stop_ids", POINTER(c_int32)), ("n_stop_ids", c_int32)]


class Stats(Structure):
    _fields_ = [("struct_size", c_uint32), ("prompt_tokens", c_int32), ("cached_tokens", c_int32),
                ("generated_tokens", c_int32), ("finish", c_int32), ("seed", c_uint64), ("prompt_seconds", c_double),
                ("decode_seconds", c_double), ("tokens_per_window", c_double), ("resident_experts", c_int32),
                ("total_experts", c_int32), ("expert_hits", c_uint64), ("expert_misses", c_uint64)]


ON_TOKENS = CFUNCTYPE(c_int, c_void_p, POINTER(c_int32), c_int32)
ON_PROGRESS = CFUNCTYPE(c_int, c_void_p, c_int32, c_int32)


class EngineError(Exception):
    pass


def _bytes(path):
    return None if path is None else str(path).encode()


class Engine:
    """One loaded model. open() takes tens of seconds; generate() blocks until the request ends."""

    def __init__(self, library, model, draft=None, draft_vocab=None, draft_head_rows=0, placement=None,
                 context=16384, prefill_columns=4096, lend_kv=False, swaps_per_window=0):
        lib = ctypes.CDLL(str(library))
        lib.torch_engine_abi_version.restype = c_uint32
        if lib.torch_engine_abi_version() != ABI_VERSION:
            raise EngineError(f"{library} has C API version {lib.torch_engine_abi_version()}, this server needs "
                              f"{ABI_VERSION}: install the server and the library from the same release.")
        lib.torch_engine_open.argtypes = [POINTER(Options), POINTER(c_void_p), c_char_p, ctypes.c_size_t]
        lib.torch_engine_generate.argtypes = [c_void_p, POINTER(c_int32), c_int32, c_int32, POINTER(Params),
                                              ON_TOKENS, ON_PROGRESS, c_void_p, POINTER(Stats)]
        lib.torch_engine_cancel.argtypes = [c_void_p]
        lib.torch_engine_cancel.restype = None
        lib.torch_engine_reset.argtypes = [c_void_p]
        lib.torch_engine_error.argtypes = [c_void_p]
        lib.torch_engine_error.restype = c_char_p
        lib.torch_engine_close.argtypes = [c_void_p]
        lib.torch_engine_close.restype = None
        self._lib = lib
        self.has_draft = draft is not None
        self.context = context
        self._keep = [_bytes(model), _bytes(draft), _bytes(draft_vocab), _bytes(placement)]  # alive while opening
        o = Options(ctypes.sizeof(Options), self._keep[0], self._keep[1], self._keep[2], draft_head_rows, self._keep[3],
                    context, prefill_columns, int(lend_kv), swaps_per_window)
        handle, error = c_void_p(), ctypes.create_string_buffer(1024)
        status = lib.torch_engine_open(ctypes.byref(o), ctypes.byref(handle), error, len(error))
        if status != OK or not handle.value:
            raise EngineError(error.value.decode(errors="replace") or f"torch_engine_open returned status {status}.")
        self._handle = handle

    def generate(self, ids, stable, params, on_tokens, on_progress):
        """Runs one request. on_tokens(list of ids) and on_progress(fed, prompt_tokens) return True to stop it.
        Returns (status, Stats, the first exception a callback raised or None)."""
        raised = []

        def tokens(_, p, n):
            try:
                return 1 if raised or on_tokens(p[:n]) else 0
            except BaseException as e:  # noqa: BLE001 - kept and reported; never into C
                raised.append(e)
                return 1

        def progress(_, fed, total):
            try:
                return 1 if raised or on_progress(fed, total) else 0
            except BaseException as e:  # noqa: BLE001
                raised.append(e)
                return 1

        stop = (c_int32 * max(1, len(params.get("stop_ids", ()))))(*params.get("stop_ids", ()))
        p = Params(ctypes.sizeof(Params), params["max_tokens"], int(params.get("speculative", False)),
                   int(params.get("sample", False)), params.get("temperature", 1.0), params.get("top_k", 0),
                   params.get("top_p", 1.0), params.get("min_p", 0.0), params.get("repetition_penalty", 1.0),
                   params.get("presence_penalty", 0.0), params.get("frequency_penalty", 0.0),
                   int(params.get("seed") is not None), params.get("seed") or 0, stop, len(params.get("stop_ids", ())))
        s = Stats(ctypes.sizeof(Stats))
        array = (c_int32 * len(ids))(*ids)
        status = self._lib.torch_engine_generate(self._handle, array, len(ids), stable, ctypes.byref(p),
                                                 ON_TOKENS(tokens), ON_PROGRESS(progress), None, ctypes.byref(s))
        return status, s, (raised[0] if raised else None)

    def error(self):
        return (self._lib.torch_engine_error(self._handle) or b"").decode(errors="replace")

    def cancel(self):
        """From any thread: the running request stops before its next window or chunk."""
        self._lib.torch_engine_cancel(self._handle)

    def reset(self):
        if self._lib.torch_engine_reset(self._handle) != OK:
            raise EngineError(self.error())

    def close(self):
        if self._handle:
            self._lib.torch_engine_close(self._handle)
            self._handle = None

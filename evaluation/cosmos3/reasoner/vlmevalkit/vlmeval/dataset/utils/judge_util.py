import atexit
import os
import threading

from vlmeval.smp import get_logger, load_env

INTERNAL = os.environ.get('INTERNAL', 0)


class JudgeTracker:
    """Wraps judge model to count and log extraction failures.

    Two tracking levels:
    - Per-call: writes to JUDGE_STATS_FILE on every failed generate() call (debug/visibility)
    - Per-item: writes to JUDGE_ITEM_STATS_FILE once per failed item, deduplicated (authoritative)

    Per-item dedup detects item boundaries by prompt hash changes within each thread.
    Digit codes: 1 = recoverable (rate limit, timeout, etc.), 2 = content_filter (non-recoverable).
    Thread-safe: shared dict + lock. Process-safe: O_APPEND atomic writes + pickle support.
    """
    _REASON_LABELS = {1: 'recoverable', 2: 'content_filter'}  # noqa: RUF012

    def __init__(self, model):
        self._model = model
        self._logger = get_logger('JudgeTracker')
        self._stats_file = os.environ.get('JUDGE_STATS_FILE')
        self._item_file = os.environ.get('JUDGE_ITEM_STATS_FILE')
        self._lock = threading.Lock()
        self._thread_state = {}  # thread_id -> {ph, failed, cf}
        atexit.register(self._flush_all)

    def _state(self):
        """Get or create per-thread tracking state."""
        tid = threading.get_ident()
        with self._lock:
            if tid not in self._thread_state:
                self._thread_state[tid] = {'ph': None, 'failed': False, 'cf': False}
            return self._thread_state[tid]

    def generate(self, *args, **kwargs):
        st = self._state()
        ph = hash(str(args[0])) if args else 0

        # New item detected (prompt changed) — finalize previous item
        if ph != st['ph']:
            self._finalize_item(st)
            st['ph'] = ph
            st['failed'] = False
            st['cf'] = False

        result = self._model.generate(*args, **kwargs)

        if result is None or self._model.fail_msg in str(result):
            reason = getattr(getattr(self._model, '_fail_reason_local', None), 'reason', 1)
            label = self._REASON_LABELS.get(reason, 'recoverable')
            self._logger.error(f'⚠️  JUDGE_CALL_FAILED ({label})')
            # Per-call stats (debug/visibility)
            if self._stats_file:
                try:
                    with open(self._stats_file, 'a') as f:
                        f.write(f'{reason}\n')
                except Exception:
                    pass
            # Per-item tracking (deduped by prompt hash)
            st['failed'] = True
            if reason == 2:
                st['cf'] = True
        else:
            # Success — item recovered on this retry
            st['failed'] = False
            st['cf'] = False

        return result

    def _finalize_item(self, st):
        """Write per-item stat for a completed item that ended in failure."""
        if not st.get('failed'):
            return
        reason = 2 if st.get('cf') else 1
        if self._item_file:
            try:
                with open(self._item_file, 'a') as f:
                    f.write(f'{reason}\n')
            except Exception:
                pass

    def _flush_all(self):
        """Flush all threads' last items at process exit."""
        with self._lock:
            for st in self._thread_state.values():
                self._finalize_item(st)
                st['failed'] = False

    def __getattr__(self, name):
        # Guard against infinite recursion during pickle deserialization:
        # __getattr__ is called before __init__ sets _model, so self._model
        # would trigger __getattr__ again. Raise AttributeError to break the cycle.
        if '_model' not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._model, name)

    def __getstate__(self):
        """Pickle support for ProcessPoolExecutor."""
        state = self.__dict__.copy()
        state.pop('_lock', None)
        state['_thread_state'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        atexit.register(self._flush_all)


def build_judge(**kwargs):
    from vlmeval.api import HFChatModel, OpenAIWrapper, SiliconFlowAPI
    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    load_env()
    LOCAL_LLM = os.environ.get('LOCAL_LLM', None)
    if LOCAL_LLM is None:
        # Check if using NVIDIA API
        using_nvidia_api = 'inference-api.nvidia.com' in os.environ.get('OPENAI_API_BASE', '')

        if using_nvidia_api:
            # NVIDIA API requires azure/openai/ prefix for GPT models
            model_map = {
                'gpt-4-turbo': 'azure/openai/gpt-4',
                'gpt-4-0613': 'azure/openai/gpt-4',
                'gpt-4-0125': 'azure/openai/gpt-4',
                'gpt-4-0409': 'azure/openai/gpt-4',
                'chatgpt-1106': 'openai/openai/gpt-3.5-turbo',
                'chatgpt-0125': 'openai/openai/gpt-3.5-turbo',
                'gpt-4o': 'azure/openai/gpt-4o',
                'gpt-4o-0806': 'azure/openai/gpt-4o',
                'gpt-4o-1120': 'azure/openai/gpt-4o',
                'gpt-4o-mini': 'azure/openai/gpt-4o-mini',
                'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
                'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
                'deepseek': 'deepseek-ai/DeepSeek-V3',
                'llama31-8b': 'meta-llama/Llama-3.1-8B-Instruct',
                'gemini-3-flash-preview': 'gcp/google/gemini-3-flash-preview',
                'gemini-3-pro-preview': 'gcp/google/gemini-3.1-pro-preview',
            }
        else:
            # Original OpenAI API model mapping
            model_map = {
                'gpt-4-turbo': 'gpt-4-1106-preview',
                'gpt-4-0613': 'gpt-4-0613',
                'gpt-4-0125': 'gpt-4-0125-preview',
                'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
                'chatgpt-1106': 'gpt-3.5-turbo-1106',
                'chatgpt-0125': 'gpt-3.5-turbo-0125',
                'gpt-4o': 'gpt-4o-2024-05-13',
                'gpt-4o-0806': 'gpt-4o-2024-08-06',
                'gpt-4o-1120': 'gpt-4o-2024-11-20',
                'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
                'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
                'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
                'deepseek': 'deepseek-ai/DeepSeek-V3',
                'llama31-8b': 'meta-llama/Llama-3.1-8B-Instruct',
            }
        model_version = model_map[model] if model in model_map else model
    else:
        model_version = LOCAL_LLM

    if model in ['qwen-7b', 'qwen-72b', 'deepseek']:
        model = SiliconFlowAPI(model_version, **kwargs)
    elif model == 'llama31-8b':
        model = HFChatModel(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    model = JudgeTracker(model)
    return model


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You can see the specific error if the API call fails.
"""

from vllm import LLM, SamplingParams

class VLLMRunner:
    def __init__(self, model_id, tensor_parallel_size=1, max_model_len=4096):
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len
        )

    def generate(self, prompts, temperature=0.0, max_tokens=256):
        params = SamplingParams(stop=['\n\nQuestion:', '\nQuestion:'], 
            temperature=0.0,
            max_tokens=256)
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

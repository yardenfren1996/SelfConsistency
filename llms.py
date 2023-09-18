import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    AutoTokenizer, AutoModelForSeq2SeqLM


class Pegasus:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name, cache_dir='./models')
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name, cache_dir='./models').to(device)

    def summarize(self, document, no_generations=10, no_seeds=5):
        try:
            batch = self.tokenizer(document, truncation=True, padding="longest", return_tensors="pt").to(self.device)
            outputs = []
            for seed in range(no_seeds):
                torch.manual_seed(seed)
                translated = self.model.generate(**batch, num_return_sequences=no_generations,
                                                 num_beams=no_generations,
                                                 do_sample=True)
                tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
                outputs.extend(tgt_text)
            return outputs
        except Exception as e:
            raise type(e)(f'failed to summarize, due to: {e}')


class Llama_2:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir='./models')

        self.model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./models',  # load_in_8bit=True,
                                                      device_map='auto')  # , torch_dtype=torch.float16)
        # load_in_4bit=True, trust_remote_code=True)
        self.model.eval()

    def summarize(self, document, no_generations=10, no_seeds=5):
        try:
            eval_prompt = f"""
Summarize the following:
{document}
---
Summary:
"""
            outputs = []
            with torch.no_grad():
                for seed in range(no_seeds):
                    torch.manual_seed(seed)
                    model_input = self.tokenizer(eval_prompt, return_tensors="pt").to("cuda")
                    output = self.model.generate(**model_input, top_k=10,  # do_sample=True,
                                                 num_return_sequences=no_generations, max_new_tokens=100)
                    tgt_text = [self.tokenizer.decode(out, skip_special_tokens=True).split('Summary:')[1].strip()
                                for out
                                in
                                output]
                    outputs.extend(tgt_text)
            return outputs
        except Exception as e:
            raise type(e)(f'failed to summarize, due to: {e}')

    def paraphrase(self, sentence, num_return_sequences=1):
        try:
            eval_prompt = f"""
Paraphrase the following:
{sentence}
---
Paraphrased:
"""
            with torch.no_grad():
                model_input = self.tokenizer(eval_prompt, return_tensors="pt").to("cuda")
                output = self.model.generate(**model_input, top_k=10, do_sample=True,
                                             num_return_sequences=num_return_sequences, max_new_tokens=100)

                outputs = [self.tokenizer.decode(out, skip_special_tokens=True).split('Paraphrased:')[1].strip() for out
                           in
                           output]
            return outputs
        except Exception as e:
            raise type(e)(f'failed to paraphrase, due to: {e}')


class Flan_T5:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./models', device_map='auto')
        self.model.eval()

    def summarize(self, document, no_generations=10, no_seeds=5):
        try:
            eval_prompt = f"Summarize: {document}"
            outputs = []
            with torch.no_grad():
                for seed in range(no_seeds):
                    torch.manual_seed(seed)
                    input_ids = self.tokenizer(eval_prompt, truncation=True, return_tensors="pt").input_ids.to(
                        self.device)
                    output = self.model.generate(input_ids, top_k=10, do_sample=True,
                                                 num_return_sequences=no_generations, max_new_tokens=100)
                    tgt_text = [self.tokenizer.decode(out, skip_special_tokens=True) for out in output]
                    outputs.extend(tgt_text)
            return outputs
        except Exception as e:
            raise type(e)(f'failed to summarize, due to: {e}')


class Paraphraser:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./models').to(self.device)
        self.model.eval()

    def paraphrase(
            self,
            sentence,
            num_beams=5,
            num_beam_groups=5,
            num_return_sequences=5,
            repetition_penalty=10.0,
            diversity_penalty=3.0,
            no_repeat_ngram_size=2,
            temperature=0.7,
    ):
        try:
            with torch.no_grad():
                input_ids = self.tokenizer(
                    f'paraphrase: {sentence}',
                    return_tensors="pt", padding="longest",
                    truncation=True,
                ).input_ids.to(self.device)

                outputs = self.model.generate(
                    input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
                    num_beams=num_beams, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty,
                    max_new_tokens=100
                )

                res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                return res
        except Exception as e:
            raise type(e)(f'failed to paraphrase, due to: {e}')

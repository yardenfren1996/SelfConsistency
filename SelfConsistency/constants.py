from SelfConsistency.llms import Pegasus, Llama_2, Flan_T5

SUMMARY_MODEL_CONFIG = {'pegasus': {'cls': Pegasus, 'config': {'model_name': "google/pegasus-xsum"}},
          'llama_2': {'cls': Llama_2, 'config': {'model_name': "meta-llama/Llama-2-7b-chat-hf"}},
          'flan_t5_xl': {'cls': Flan_T5, 'config': {'model_name': "google/flan-t5-xl"}},
          'flan_t5': {'cls': Flan_T5, 'config': {'model_name': "google/flan-t5-large"}}}
import importlib
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModel
from test_bed_local.serve.model_info.model_execute_info.dpt_beit_large_512 import SplitDPTModel
from transformers import BertForQuestionAnswering, BertConfig
import time
from transformers import AutoConfig, GenerationConfig
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
)
from accelerate import init_empty_weights
# from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel,CLIPConfig

from test_bed_local.serve.model_info.models.llama.generation import Llama
from test_bed_local.serve.utils.data_structure import is_llm

tokenizer = None
model = None

def load_empty_model_and_tokenizer(model_name,root_path,device_map,block_num):
    if is_llm(model_name):
        return Llama.build_empty_model_and_tokenizer(
            device_map=device_map,
            block_num=block_num,
            ckpt_dir=f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/{model_name}/',
            tokenizer_path=f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/tokenizer.model',
            max_seq_len=256,
            max_batch_size=6,
        )
    elif model_name == 'bertqa':
        config = AutoConfig.from_pretrained(
            'bert-large-uncased-whole-word-masking-finetuned-squad', trust_remote_code=True
        )
        print('config',time.time()-start)
        hf_model_class = 'AutoModel'
        start = time.time()
        with init_empty_weights():
            module = importlib.import_module("transformers")
            _class = getattr(module, hf_model_class)
            model = _class.from_config(config, trust_remote_code=True).to(config.torch_dtype)
        model.tie_weights()
        return (model,None)
    elif model_name == 'clip-vit-large-patch14':
        config = AutoConfig.from_pretrained(
            'openai/clip-vit-large-patch14', trust_remote_code=True
        )
        print('config',time.time()-start)
        hf_model_class = 'AutoModel'
        start = time.time()
        with init_empty_weights():
            module = importlib.import_module("transformers")
            _class = getattr(module, hf_model_class)
            model = _class.from_config(config, trust_remote_code=True).to(config.torch_dtype)
        model.tie_weights()
        return (model,None)
    
def load_tokenizer(model_name,root_path):
    if is_llm(model_name):
        return Llama.build_tokenizer(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/tokenizer.model')
    else:
        return None
def load_model_by_name(model_name,device_map,block_num,root_path):
    global model
    global tokenizer

    if model_name == 'bertqa':
        res = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        res.eval()
        res.cuda(device_map[0])

        x = torch.cat((torch.ones((1, 512), dtype=int).view(-1), torch.ones((1, 512), dtype=int).view(-1))).view(2, -1, 512).cuda(device_map[0])
        start_t = time.time()
        with torch.no_grad():
            y = res(x[0], token_type_ids=x[1])
        del x
        return res
    elif model_name == 'clip-vit-large-patch14':
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.eval()
        model.cuda(device_map[0])
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
        inputs = {key: value.cuda(device_map[0]) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        return model
    elif model_name == 'multilingual-e5-large':
        # Each input text should start with "query: " or "passage: ", even for non-English texts.
        # For tasks other than retrieval, you can simply use the "query: " prefix.
        input_texts = ['query: how much protein should a female eat',
               'query: 南瓜的家常做法',
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
               "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"]

        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
        model.eval()        
        model.cuda(device_map[0])

        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {key: value.cuda(device_map[0]) for key, value in batch_dict.items()}
        with torch.no_grad():
            outputs = model(**batch_dict)
    elif is_llm(model_name):
        model = Llama.build_model(
            device_map=device_map,
            block_num=block_num,
            ckpt_dir=f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/{model_name}/',
            tokenizer_path=f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/tokenizer.model',
            max_seq_len=256,
            max_batch_size=6,
        )
        return model
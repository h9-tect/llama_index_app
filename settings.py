import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from llama_index.core.prompts import PromptTemplate

def initialize_settings(api_key):
    # Check if a GPU is available
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        # Use a CPU configuration if no GPU is available
        quantization_config = None

    def messages_to_prompt(messages):
        prompt = ""
        for message in messages:
            if message.role == 'system':
                prompt += f"\n{message.content}</s>\n"
            elif message.role == 'user':
                prompt += f"\n{message.content}</s>\n"
            elif message.role == 'assistant':
                prompt += f"\n{message.content}</s>\n"
        if not prompt.startswith("\n"):
            prompt = "\n</s>\n" + prompt
        prompt = prompt + "\n"
        return prompt

    Settings.llm = HuggingFaceLLM(
        model_name="HuggingFaceH4/zephyr-7b-alpha",
        tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
        query_wrapper_prompt=PromptTemplate("\n</s>\n\n{query_str}</s>\n\n"),
        context_window=3900,
        max_new_tokens=256,
        model_kwargs={"quantization_config": quantization_config} if quantization_config else {},
        generate_kwargs={"do_sample": False},
        messages_to_prompt=messages_to_prompt,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def load_documents(api_key):
    return LlamaParse(result_type="markdown", api_key=api_key).load_data('/kaggle/input/dataset2/AUTOSAR_SWS_OCUDriver.pdf')

def load_documents_from_file(api_key, uploaded_file):
    with open('/tmp/temp.pdf', 'wb') as f:
        f.write(uploaded_file.read())
    return LlamaParse(result_type="markdown", api_key=api_key).load_data('/tmp/temp.pdf')

def setup_node_parser(node_parser, documents):
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    raw_index = VectorStoreIndex.from_documents(documents)
    index_with_obj = VectorStoreIndex(nodes=base_nodes+objects)
    return nodes, base_nodes, objects, raw_index, index_with_obj

def setup_query_engines(nodes, base_nodes, objects, raw_index, index_with_obj):
    from llama_index.core.retrievers import RecursiveRetriever
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    from llama_index.core.query_engine import RetrieverQueryEngine

    reranker = FlagEmbeddingReranker(
        top_n=5,
        model="BAAI/bge-reranker-large",
    )

    index_with_obj_query_engine = index_with_obj.as_query_engine(
        similarity_top_k=15, 
        node_postprocessors=[reranker], 
        verbose=False
    )

    raw_query_engine = raw_index.as_query_engine(similarity_top_k=15, node_postprocessors=[reranker])

    index = VectorStoreIndex(nodes=base_nodes)
    index_ret = index.as_retriever(top_k=15)
    recursive_index = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": index_ret},
        node_dict={node.index_id: node for node in objects},
        verbose=False,
    )

    recursive_query_engine = RetrieverQueryEngine.from_args(recursive_index, node_postprocessors=[reranker], verbose=False)

    return raw_query_engine, index_with_obj_query_engine, recursive_query_engine

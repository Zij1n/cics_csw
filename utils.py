import os
os.environ["HF_HOME"] = "/data2/zijin/.cache/huggingface"
import pkuseg
import re
import random
import torch
import os 
from huggingface_hub import login

login()
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer, AutoModelForCausalLM
seg = pkuseg.pkuseg()


# Load translation models
translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Load Llama model and tokenizer
import transformers
import torch
# Initialize the pipeline
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
llama_tokenizer = pipeline.tokenizer
llama_model = pipeline.model
llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id

def translate(word, source_lang, target_lang):
    translation_tokenizer.src_lang = source_lang
    encoded_text = translation_tokenizer(word, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded_text,
        forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang)
    )
    return translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].lower()


# Word segmentation for mixed-language sentences
def mixed_language_word_seg(sentence):
    tokenized_words = []
    for part in re.split(r'([\u4e00-\u9fa5]+)', sentence):
        if re.match(r'[\u4e00-\u9fa5]+', part):  # Match Chinese
            tokenized_words.extend(seg.cut(part))
        else:  # Match other characters
            tokenized_words.extend(part.strip().split())
    return [word for word in tokenized_words if word]


# Identify the language of a word
def language_identifier(word):
    if re.search(r'[\u4e00-\u9fa5]', word):
        return "zh"  # Chinese
    elif re.search(r'[A-Za-z]', word):
        return "en"  # English
    return None


# Randomly translate a mixed-language sentence
def random_translate(sentence_words, k):
    selected_indices = random.sample(range(len(sentence_words)), min(k, len(sentence_words)))
    translated_sentence = []
    for i, word in enumerate(sentence_words):
        lang = language_identifier(word)
        if i in selected_indices:
            translated_sentence.append(
                translate(word, "zh", "en") if lang == "zh" else translate(word, "en", "zh")
            )
        else:
            translated_sentence.append(
                translate(word, "en", "zh") if lang == "en" else word
            )
    return translated_sentence


def compare_sentences(original, translated):
    # Define the prompt in the role-based format
    prompt = [
        {"role": "system", "content": "You are a helpful, concise, and polite assistant. Respond to the user's query in a clear and informative manner."},
        {"role": "user", "content": f"""
Which of the following code-switched sentences is more natural to you?

1: {original}

2: {translated}

Respond with your choice in the format: (response:<1 or 2>).
        """},
    ]
    
    # Generate the response
    generation = pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=50,
        repetition_penalty=1.2,
        pad_token_id=pipeline.tokenizer.eos_token_id,
    )
    
    # Extract the assistant's response
    response_content = generation[0].get("generated_text", "")[-1]["content"]
    # print("model's resposne:")
    # print(response_content)
    # print(type(response_content))
    # print("----------------")
    # Parse the response to find the chosen option
    import re
    match = re.search(r'\(response: (\d)\)', response_content)
    return int(match.group(1)) if match else None

def compute_token_nll(sentence):
    sentence_tokens = llama_tokenizer(sentence, return_tensors="pt").input_ids
    nll_per_token = []

    with torch.no_grad():
        # Compute model outputs for the full sentence
        outputs = llama_model(input_ids=sentence_tokens, labels=sentence_tokens)
        logits = outputs.logits  # Logits for each token in the sequence

    # Compute the NLL for each token
    token_labels = sentence_tokens.squeeze(0)
    num_tokens = token_labels.size(0)

    # Use CrossEntropyLoss to compute token-level NLL
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    for i in range(num_tokens):
        # Extract logits for the current token
        token_logits = logits[:, i, :].squeeze(0)
        token_label = token_labels[i]

        # Calculate NLL for the token
        token_nll = loss_fn(token_logits.unsqueeze(0), token_label.unsqueeze(0)).item()
        nll_per_token.append((llama_tokenizer.decode([token_label.item()]), token_nll))

    return nll_per_token

def compute_word_nll(sentence, segmenter):
    """
    Compute the NLL for each word in a segmented sentence, excluding the start-of-sentence token.

    Args:
        sentence (str): The input sentence.
        segmenter (callable): A function to segment the input sentence into words.

    Returns:
        list: A list of tuples containing each word and its corresponding NLL.
    """
    # Segment the sentence into words
    word_segments = segmenter(sentence)

    # Tokenize the entire sentence
    sentence_tokens = llama_tokenizer(sentence, return_tensors="pt").input_ids.squeeze(0)

    with torch.no_grad():
        # Compute model outputs for the full sentence
        outputs = llama_model(input_ids=sentence_tokens.unsqueeze(0), labels=sentence_tokens.unsqueeze(0))
        logits = outputs.logits  # Logits for each token in the sequence

    # Token labels and logits (excluding the start-of-sentence token)
    token_labels = sentence_tokens[1:]  # Exclude the start-of-sentence token
    logits = logits[0, 1:, :]  # Adjust logits to match the tokens (excluding SOS)

    # Use CrossEntropyLoss to compute token-level NLL
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    token_nlls = loss_fn(logits, token_labels).tolist()  # List of token-level NLLs

    # Queue for tokens and their NLLs
    token_queue = [(llama_tokenizer.decode([token_labels[i].item()]), token_nlls[i]) for i in range(len(token_labels))]

    # Map tokens to words
    word_nlls = []
    for word in word_segments:
        word_nll = 0
        matched_tokens = []

        # Dequeue tokens and accumulate NLLs until the full word is matched
        while len(matched_tokens) < len(word):
            token, token_nll = token_queue.pop(0)
            matched_tokens.append(token)
            word_nll += token_nll

            # Check if the dequeued tokens match the start or full word
            matched_word = "".join(matched_tokens).replace(" ", "")
            if matched_word == word:
                break
            if len(matched_word)>len(word):
                return ["error: a token is made of different word"]

        # Append the word and its accumulated NLL
        word_nlls.append((word, word_nll))

    return word_nlls


# # Sentence
# sentence = "快要期末考试了他可能觉得非常stress非常nervous"

# # Compute word NLLs
# word_nlls = compute_word_nll(sentence, mixed_language_word_seg)

# # Print results
# for word, nll in word_nlls:
#     print(f"Word: {word}, NLL: {nll}")


import re

def clean_mixed_language_sentence(sentence):
    # Remove all punctuation
    sentence = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', sentence)

    # Remove spaces between Chinese characters and between Chinese and English characters
    sentence = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', sentence)  # Chinese-Chinese
    sentence = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[A-Za-z])', '', sentence)        # Chinese-English
    sentence = re.sub(r'(?<=[A-Za-z])\s+(?=[\u4e00-\u9fa5])', '', sentence)        # English-Chinese

    # Convert to lowercase
    sentence = sentence.lower()

    return sentence

def calculate_sentence_perplexity(sentence):
    # Tokenize the input sentence and prepare inputs for the model
    inputs = llama_tokenizer(sentence, return_tensors="pt")
    input_ids = inputs.input_ids

    # Compute the outputs with labels set to the same input_ids
    with torch.no_grad():
        outputs = llama_model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss  # Cross-entropy loss

    # Calculate perplexity
    perplexity = torch.exp(loss).item()
    return perplexity

def count_english_words(sentence):
    return sum(1 for word in sentence.split() if language_identifier(word) == "en")

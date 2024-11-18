# Are multilingual LLMs like multilingual humans? Insights from code-switching  
Xuanyi (Jessica) Chen, Zijin Hu, *New York University*   
This repo contains code and datasets for CICS course project  



### Generating Dataset
Run `python make_dataset.py --input_csv="path/to/input.csv" --output_csv="path/to/output.csv" --n_translations=5`  

### Running Stats
See `stats.ipynb` for detail 

<!-- ### Data Columns

This dictionary structure stores the results of processing sentences for code-switching analysis, recording comparisons between original and transformed sentences along with associated metrics.

#### Fields:
1. **`original`**:  
   - **Type**: `str`  
   - **Description**: The cleaned version of the original sentence after segmentation and preprocessing.

2. **`transformed`**:  
   - **Type**: `str`  
   - **Description**: The cleaned version of the transformed sentence, where a subset of words in the original sentence have been randomly translated to the other language.

3. **`llama_preference`**:  
   - **Type**: `int`  
   - **Description**: The choice made by the Llama model comparing the naturalness of the original and transformed sentences.  
   - **Values**:
     - `1`: Indicates the transformed sentence is preferred.
     - `2`: Indicates the original sentence is preferred.  
     
4. **`original_word_nll`**:  
   - **Type**: `list of tuples` or `str`  
   - **Description**: A list of tuples, where each tuple contains a word from the original sentence and its negative log-likelihood (NLL) as computed by the model.  
   - **Error Handling**: If an error occurs during computation, this field may contain a string indicating the error instead of the expected list of tuples.  

5. **`original_token_nll`**:  
   - **Type**: `list of tuples`  
   - **Description**: A list of tuples, where each tuple contains a token from the original sentence and its NLL. Tokens are finer-grained than words.

6. **`original_perplexity`**:  
   - **Type**: `float`  
   - **Description**: The perplexity of the original sentence as computed by the Llama model. Lower perplexity indicates a more predictable and natural sentence.

7. **`transformed_word_nll`**:  
   - **Type**: `list of tuples` or `str`  
   - **Description**: A list of tuples, where each tuple contains a word from the transformed sentence and its NLL.  
   - **Error Handling**: If an error occurs during computation, this field may contain a string indicating the error instead of the expected list of tuples.

8. **`transformed_token_nll`**:  
   - **Type**: `list of tuples`  
   - **Description**: A list of tuples, where each tuple contains a token from the transformed sentence and its NLL.

9. **`transformed_perplexity`**:  
   - **Type**: `float`  
   - **Description**: The perplexity of the transformed sentence as computed by the Llama model.

10. **`original_number_of_code_switch`**:  
    - **Type**: `int`  
    - **Description**: The number of English words in the original sentence, representing the frequency of code-switching in the original text.

11. **`transformed_number_of_code_switch`**:  
    - **Type**: `int`  
    - **Description**: The number of English words in the transformed sentence, representing the frequency of code-switching after the transformation. -->




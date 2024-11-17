import pandas as pd
from tqdm import tqdm
import fire
from utils import (
    mixed_language_word_seg,
    random_translate,
    clean_mixed_language_sentence,
    compare_sentences,
    compute_token_nll,
    compute_word_nll,
    calculate_sentence_perplexity,
    count_english_words,
)

def process_sentences(input_csv="data/ASDEND_filtered.csv", output_csv="processed_sentences.csv", n_translations=5):
    """
    Process sentences for code-switching analysis and save results to a CSV file.

    Args:
        input_csv (str): Path to the input CSV file (default: 'data/ASDEND_filtered.csv').
        output_csv (str): Path to the output CSV file (default: 'processed_sentences.csv').
        n_translations (int): Number of translations to generate per sentence (default: 5).
    """
    output_data = []

    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Process each sentence
    for sentence in tqdm(df["sentence"]):
        original_sentence = " ".join(mixed_language_word_seg(sentence))
        cleaned_original = clean_mixed_language_sentence(original_sentence)

        # Compute properties of the original sentence
        original_token_nll = compute_token_nll(cleaned_original)
        try:
            original_word_nll = compute_word_nll(cleaned_original, mixed_language_word_seg)
        except Exception as e:
            original_word_nll = f"Error: {str(e)}"
        original_perplexity = calculate_sentence_perplexity(cleaned_original)

        # Generate and process n translated sentences
        for _ in range(n_translations):
            translated_sentence = " ".join(
                random_translate(mixed_language_word_seg(sentence), 3)
            )
            cleaned_translated = clean_mixed_language_sentence(translated_sentence)

            # Compare sentences using Llama
            llama_choice = compare_sentences(cleaned_translated, cleaned_original)

            # Compute properties of the transformed sentence
            translated_token_nll = compute_token_nll(cleaned_translated)
            try:
                translated_word_nll = compute_word_nll(cleaned_translated, mixed_language_word_seg)
            except Exception as e:
                translated_word_nll = f"Error: {str(e)}"
            translated_perplexity = calculate_sentence_perplexity(cleaned_translated)
            original_number_of_code_switch = count_english_words(cleaned_original)
            transformed_number_of_code_switch = count_english_words(cleaned_translated)

            # Store results in the output list
            output_data.append(
                {
                    "original": cleaned_original,
                    "transformed": cleaned_translated,
                    "llama_preference": llama_choice,
                    "original_word_nll": original_word_nll,
                    "original_token_nll": original_token_nll,
                    "original_perplexity": original_perplexity,
                    "transformed_word_nll": translated_word_nll,
                    "transformed_token_nll": translated_token_nll,
                    "transformed_perplexity": translated_perplexity,
                    "original_number_of_code_switch": original_number_of_code_switch,
                    "transformed_number_of_code_switch": transformed_number_of_code_switch,
                }
            )

    # Convert the output to a DataFrame and save to CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)

    print(f"Processing complete. Results saved to {output_csv}.")

if __name__ == "__main__":
    fire.Fire(process_sentences)

# GPTRUN

## Overview
BotFollowupRanker is a framework for generating and evaluating automated responses using various ranking and validation techniques. It processes queries, generates follow-ups using GPT-based models, and validates responses for competition-based evaluation.

## File Breakdown

### `text_validation.py`
- **Purpose**: Processes and validates text responses, ensuring they meet constraints such as word count.
- **Inputs**: Query XML file, bot follow-up CSV files.
- **Outputs**: Processed text responses.

### `competition_chatgpt_google.py`
- **Purpose**: Runs competitive evaluation of GPT-based responses.
- **Inputs**: Config settings, bot follow-up data.
- **Outputs**: Ranked results.

### `config.py`
- **Purpose**: Stores configuration parameters, including model settings and competition parameters.
- **Inputs**: User-defined parameters.
- **Outputs**: Configurable settings for execution.

### `create_bot_followup_file.py`
- **Purpose**: Generates structured bot follow-up data.
- **Inputs**: Competition data files (`t_data.csv`, `g_data.csv`).
- **Outputs**: CSV files containing bot follow-ups.

## Installation & Usage
1. **Activate Conda Environment:**
   ```sh
   conda activate GPTRUN
   ```
2. **Ensure Configuration Files Are Set:**
   - Update `config.py` with the desired parameters.
3. **Run Bot Follow-up Generation:**
   ```sh
   python create_bot_followup_file.py
   ```
4. **Validate and Process Texts:**
   ```sh
   python text_validation.py
   ```
5. **Run Competition Evaluation:**
   ```sh
   python competition_chatgpt_google.py
   ```

## Dependencies
- Python 3.x
- OpenAI API
- NumPy, Pandas, TQDM, NLTK
- Tiktoken (for token handling)

## Notes
- Ensure `bot_followup_{cp}.csv` and required data files exist before execution.
- Modify `config.py` to adjust parameters for different ranking strategies.
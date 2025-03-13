import streamlit as st
import pandas as pd
from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer
import gc

# Cache model loading to prevent reloading
@st.cache_resource
def load_model(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # Do this once during loading
    return model, tokenizer

# Optimized prediction and explanation function
def predict_sql(model, tokenizer, prompt, mode="prediction"):
    """
    Generate either a SQL query or an explanation based on the mode.
    """
    if mode == "prediction":
        # Generate SQL query
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200, use_cache=True)
        results = tokenizer.batch_decode(outputs)
        return results[0]
    elif mode == "explanation":
        # Generate explanation for the SQL query
        explanation_prompt = f"Explain the following SQL query: {prompt}"
        inputs = tokenizer(explanation_prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200, use_cache=True)
        results = tokenizer.batch_decode(outputs)
        return results[0]

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Load validation data from CSV and cache it
@st.cache_data
def load_validation_data():
    return pd.read_csv("/content/drive/MyDrive/Tanmay-validation-data/validation_data.csv")

# Streamlit UI
def main():
    st.title("GenAI Course: Text-to-SQL Model Comparison")

    # Load validation data (cached)
    validation_data = load_validation_data()

    # Model selection dropdown
    model_options = {
        "Base Mistral ": "/content/drive/MyDrive/Tanmay-models/mistral-7b-base",
        "Fine-Tuned Mistral ": "/content/drive/MyDrive/Tanmay-models/mistral-7b-fine-tuned",
        "Base Llama3 ": "/content/drive/MyDrive/Tanmay-models/llama-3-8b-base",
        "Fine-Tuned Llama3 ": "/content/drive/MyDrive/Tanmay-models/llama-3-8b-fine-tuned",
    }
    
    # Initialize session state for model caching
    if "current_model_name" not in st.session_state:
        st.session_state.current_model_name = None
        st.session_state.current_model = None
        st.session_state.current_tokenizer = None
    
    # Use a form for selection to batch UI interactions
    with st.form("selection_form"):
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        
        # Domain selection
        domains = validation_data["domain"].unique().tolist()
        selected_domain = st.selectbox("Select Domain", domains)
        
        # Pre-filter data for SQL complexity
        filtered_data = validation_data[validation_data["domain"] == selected_domain]
        sql_complexities = filtered_data["sql_complexity"].unique().tolist()
        selected_complexity = st.selectbox("Select SQL Complexity", sql_complexities)
        
        # Filter for queries
        filtered_queries = filtered_data[filtered_data["sql_complexity"] == selected_complexity]
        query_options = filtered_queries["text"].tolist()
        selected_query = st.selectbox("Select an example query", query_options)
        
        # Add the "Run Prediction" button inside the form
        submitted = st.form_submit_button("Run Prediction")
    
    if submitted:
        # Get the model path
        model_path = model_options[selected_model]
        
        # Load model only when selection changes
        if selected_model != st.session_state.current_model_name:
            clear_gpu_memory()  # Clear previous model from memory
            with st.spinner(f"Loading {selected_model}..."):
                st.session_state.current_model, st.session_state.current_tokenizer = load_model(model_path)
                st.session_state.current_model_name = selected_model
        
        # Get selected example's data
        example_data = filtered_queries[filtered_queries["text"] == selected_query].iloc[0]
        text = example_data["text"]

        # Extract ground truth SQL from the text field
        input_prompt = text.split("### Response")[0]
        ground_truth_sql = text.split("### Response")[1].strip()

        # Display input prompt and ground truth
        st.subheader("Input Prompt")
        st.code(input_prompt, language="plaintext")

        st.subheader("Ground Truth SQL Query")
        st.code(ground_truth_sql, language="sql")

        # Run prediction
        with st.spinner("Generating SQL..."):
            predicted_sql = predict_sql(
                st.session_state.current_model, 
                st.session_state.current_tokenizer, 
                input_prompt,
                mode="prediction"
            )
            
            # Display predicted SQL query
            st.subheader("Predicted SQL Query")
            st.code(predicted_sql, language="sql")

            # Generate and display explanation
            with st.spinner("Generating Explanation..."):
                if "Fine-Tuned" in selected_model:
                    # Use the sql_explanation from the validation dataset
                    explanation = example_data["sql_explanation"]
                else:
                    # Use the same model to generate an explanation
                    explanation = predict_sql(
                        st.session_state.current_model, 
                        st.session_state.current_tokenizer, 
                        predicted_sql,
                        mode="explanation"
                    )
                st.subheader("Explanation of the Predicted SQL Query")
                st.write(explanation)

if __name__ == "__main__":
    main()
import pandas as pd
import openai
from textblob import TextBlob
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

# Set your OpenAI API key
OPENAI_API_KEY = "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"  
openai.api_key = OPENAI_API_KEY

# Function to load the dataset
def load_medicine_data(file_path):
    """Load the dataset from a JSONL file."""
    try:
        return pd.read_json(file_path, lines=True)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

# Function to handle missing columns by filling with default values
def handle_missing_columns(df, required_columns):
    """Ensure all required columns are present and fill missing ones with default values."""
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column: {col}. Adding it with default values.")
            # Fill missing columns with a default value
            if col == 'id':
                df[col] = 0  # Default ID (can adjust this as per the dataset)
            elif col == 'price':
                df[col] = 0.0  # Default price
            elif col == 'Is_discontinued':
                df[col] = False  # Default to not discontinued
            elif col == 'manufacturer_name':
                df[col] = "Unknown"  # Default manufacturer
            elif col == 'type':
                df[col] = "Unknown"  # Default type
            elif col == 'pack_size_label':
                df[col] = "Unknown"  # Default pack size
            elif col == 'short_composition1' or col == 'short_composition2':
                df[col] = "None"  # Default composition
            elif col == 'Supplier_Performance':
                df[col] = 0  # Default supplier performance
            elif col == 'Economic_Factor':
                df[col] = 0  # Default economic factor
            elif col == 'Transport_Status':
                df[col] = "Unknown"  # Default transport status
    return df

# Function for risk analysis using OpenAI
def analyze_risks(row):
    """
    Analyze risks for each medicine record using OpenAI API.
    Combines relevant fields into a single prompt for analysis.
    """
    try:
        description = (
            f"Given the following data for a product, calculate the risk score:\n"
            f"id: {row['id']}\n"
            f"name: {row['name']}\n"
            f"price: ₹{row['price']}\n"
            f"Is_discontinued: {row['Is_discontinued']}\n"
            f"manufacturer_name: {row['manufacturer_name']}\n"
            f"type: {row['type']}\n"
            f"pack_size_label: {row['pack_size_label']}\n"
            f"short_composition1: {row['short_composition1']}\n"
            f"short_composition2: {row['short_composition2']}\n"
            f"Supplier_Performance: {row['Supplier_Performance']}\n"
            f"Economic_Factor: {row['Economic_Factor']}\n"
            f"Transport_Status: {row['Transport_Status']}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an AI that calculates risk scores for medical products."},
                      {"role": "user", "content": description}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error in risk analysis: {str(e)}"

# Function for sentiment analysis
def get_sentiment_score(text):
    """Calculate sentiment polarity for a given text."""
    if isinstance(text, str):
        return TextBlob(text).sentiment.polarity
    return 0

# Function to process a single row (called in parallel)
def process_row(row):
    """Process sentiment and risk analysis for each row."""
    row['Sentiment'] = get_sentiment_score(row['name'])
    row['Risk Analysis'] = analyze_risks(row)
    return row

# Function to visualize sentiment data
def visualize_sentiments(df, column):
    """Visualize sentiment scores as a histogram."""
    plt.figure(figsize=(10, 6))
    df[column].hist(bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {column.capitalize()} Sentiments")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

# Main function
def main():
    # Start timing
    start_time = time.time()

    # File path to your medicine dataset
    file_path = "newjson_dataset.jsonl"  # Replace with the correct dataset path

    # Load dataset
    df = load_medicine_data(file_path)
    if df is None:
        return

    # Define the required columns
    required_columns = [
        'id', 'name', 'price', 'Is_discontinued', 'manufacturer_name', 'type',
        'pack_size_label', 'short_composition1', 'short_composition2',
        'Supplier_Performance', 'Economic_Factor', 'Transport_Status'
    ]

    # Handle missing columns
    df = handle_missing_columns(df, required_columns)

    # Parallel processing: use multiprocessing Pool to process rows concurrently
    with Pool(processes=12) as pool:  # Using 12 parallel processes
        df = pd.DataFrame(pool.map(process_row, [row for _, row in df.iterrows()]))

    # Save processed data to a new file in JSONL format
    output_file = "processed_medicine_data.jsonl"
    df.to_json(output_file, orient='records', lines=True)
    print(f"Processed data saved to {output_file}.")

    # Visualize sentiment analysis results
    visualize_sentiments(df, 'Sentiment')

    # End timing
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import os

# Load the pre-trained GPT-2 model and tokenizer with error handling
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' not found.")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return tokenizer, model

# Load the model
try:
    model_path = 'C:/Users/gvnsm/OneDrive/Desktop/manju/trained_gpt2_model'
    tokenizer, model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Set maximum input length and maximum number of new tokens for generation
MAX_INPUT_LENGTH = 512  # Increase the max length here if needed
MAX_NEW_TOKENS = 100  # Maximum number of new tokens to generate in the response

def get_chatbot_response(query):
    """Generate chatbot response based on the user's query."""
    try:
        # Tokenize the input query with proper padding and truncation
        inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=MAX_INPUT_LENGTH, padding=True)
        
        # Generate the response with a limit on the number of new tokens
        response = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=MAX_NEW_TOKENS, pad_token_id=50256)
        
        # Decode and return the response as a string
        return tokenizer.decode(response[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating response: {e}"

def extract_value(response, field):
    """Extract a value for a given field like 'Risk Score', 'Economic Factor', etc."""
    try:
        match = re.search(rf'{field}: ([\d\.]+)', response)
        if match:
            return match.group(1)  # Return the value found
        return None
    except Exception as e:
        return f"Error extracting value: {e}"

def get_medicine_details(medicine_name):
    """Query the chatbot for detailed information about the medicine."""
    query = f"Given the following data for a product, provide all the details:\nname: {medicine_name}"
    response = get_chatbot_response(query)
    
    # Return the response
    return response

# Main loop to handle user input and display responses
def main():
    print("\nWelcome to the Medication Assistance System!")
    print("Thank you for visiting. I hope you're having a wonderful day!")

    while True:
        try:
            # Get user input
            medicine_name = input("\nPlease enter the medicine name (Type 'exit' to quit): ")
            
            if medicine_name.lower() == 'exit':
                print("Exiting the system. Have a great day!")
                break

            print(f"\nThank you for choosing {medicine_name}.")
            response = get_medicine_details(medicine_name)
            
            # Check if the response contains any error
            if "Error" in response:
                print(response)
            else:
                print("\nHere is the information about the medicine:")
                print(response)

            # Ask if user wants more assistance
            print("\nIs there anything else I can help you with today? 😊 (Yes/No): ")
            continue_input = input().lower()
            if continue_input != 'yes':
                print("Exiting the system. Have a great day!")
                break
        except KeyboardInterrupt:
            print("\nExiting due to interruption. Have a great day!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

# Run the main loop
if __name__ == "__main__":
    main()

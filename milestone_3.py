import pandas as pd
import logging
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Preprocess the data
def preprocess_data(df):
    # Clean column names
    df.columns = df.columns.str.strip()

    # Standardize and clean the 'name' column
    if 'name' in df.columns:
        df['name'] = df['name'].str.strip().str.lower()

    # Convert numeric columns to appropriate types
    if 'Economic_Factor' in df.columns:
        df['Economic_Factor'] = pd.to_numeric(df['Economic_Factor'], errors='coerce')
    if 'Supplier_Performance' in df.columns:
        df['Supplier_Performance'] = pd.to_numeric(df['Supplier_Performance'], errors='coerce')

    # Handle missing data for key columns
    if 'Risk_Factor' in df.columns:
        df['Risk_Factor'] = pd.to_numeric(df['Risk_Factor'], errors='coerce').fillna(0)
    return df

# Function to load and preprocess the dataset
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df = preprocess_data(df)
        logging.info(f"Data successfully loaded and preprocessed. Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

# Generate suggestions for a product based on risk and other factors
def generate_suggestions(row: dict) -> dict:
    suggestions = []
    if row.get('Is_discontinued', False):
        suggestions.append("⚠️ Product is discontinued. Consider phasing out stock or finding alternatives.")

    if row.get('Supplier_Performance') is not None:
        supplier_score = 1 - row['Supplier_Performance'] if isinstance(row['Supplier_Performance'], float) else 0
        if supplier_score > 0.5:
            suggestions.append("⚠️ Low supplier performance. Consider switching suppliers or improving relationships.")

    if row.get('Economic_Factor') and row['Economic_Factor'] < 0.5:
        suggestions.append("⚠️ Economic factor shows risk. Adjust inventory levels and plan for delays.")

    transport_status = str(row.get('Transport_Status', '')).lower()
    if not transport_status.startswith("on time"):
        suggestions.append("⚠️ Transport status is delayed. Ensure proper logistics tracking.")

    risk_score = row.get('Risk_Factor', 0)
    if risk_score > 0.7:
        suggestions.append("🔴 High risk. Consider reducing production or increasing prices to manage demand.")
    elif risk_score > 0.4:
        suggestions.append("🟡 Moderate risk. Monitor market conditions and supply chain regularly.")
    else:
        suggestions.append("🟢 Low risk. Current operations are stable.")

    sentiment_score = 1.0 - risk_score
    return {
        'risk_score': risk_score,
        'sentiment_score': sentiment_score,
        'suggestions': suggestions
    }

# Process data for a specific product
def process_data(file_path: str, product_name: str):
    df = load_data(file_path)
    if df is None:
        return None

    if 'Risk_Factor' not in df.columns or 'name' not in df.columns:
        logging.error("'Risk_Factor' or 'name' column is missing in the DataFrame.")
        return None

    product_row = df[df['name'].str.lower() == product_name.lower()]
    if product_row.empty:
        logging.error(f"Product '{product_name}' not found.")
        return None

    product_details = product_row.iloc[0].to_dict()
    return generate_suggestions(product_details)

# GUI Application
class RiskCalculatorApp:
    def __init__(self, root, file_path):
        self.root = root
        self.file_path = file_path
        self.root.title("Product Risk Calculator")
        self.root.geometry("700x500")
        self.root.configure(bg="#F5F5F5")

        # Title Label
        self.label_title = tk.Label(
            root, text="🌟 Product Risk Calculator 🌟", font=("Arial", 24, "bold"),
            bg="#4A7C59", fg="white", padx=10, pady=10
        )
        self.label_title.pack(fill="x")

        # Input Frame
        frame = tk.Frame(root, bg="#F5F5F5")
        frame.pack(pady=20)

        self.label_product_name = tk.Label(
            frame, text="Enter Product Name:", font=("Arial", 14),
            bg="#F5F5F5"
        )
        self.label_product_name.grid(row=0, column=0, padx=10, pady=10)

        self.entry_product_name = tk.Entry(frame, width=40, font=("Arial", 14), bg="#e0e0e0")
        self.entry_product_name.grid(row=0, column=1, padx=10, pady=10)

        self.button_calculate = tk.Button(
            frame, text="Calculate Risk", font=("Arial", 14, "bold"),
            bg="#4A7C59", fg="white", command=self.calculate_risk
        )
        self.button_calculate.grid(row=1, column=1, pady=20)

        # Result Area
        self.result_frame = tk.Frame(root, bg="#F5F5F5")
        self.result_frame.pack(pady=10, fill="both", expand=True)

        self.result_label = tk.Label(
            self.result_frame, text="Risk Score: ", font=("Arial", 14), bg="#F5F5F5",
            anchor="w"
        )
        self.result_label.pack(fill="x", padx=20, pady=5)

        self.sentiment_label = tk.Label(
            self.result_frame, text="Sentiment Score: ", font=("Arial", 14), bg="#F5F5F5",
            anchor="w"
        )
        self.sentiment_label.pack(fill="x", padx=20, pady=5)

        self.suggestions_label = tk.Label(
            self.result_frame, text="Suggestions: ", font=("Arial", 14), bg="#F5F5F5",
            anchor="w", justify="left"
        )
        self.suggestions_label.pack(fill="x", padx=20, pady=5)

    def calculate_risk(self):
        product_name = self.entry_product_name.get()
        if not product_name:
            messagebox.showerror("Error", "Please enter a product name.")
            return

        result = process_data(self.file_path, product_name)
        if result is None:
            messagebox.showerror(
                "Error", "Product not found in dataset.\nEnsure the name matches exactly or check the dataset."
            )
            return

        # Update the GUI with results
        self.result_label.config(text=f"Risk Score: {result['risk_score']:.2f}")
        self.sentiment_label.config(text=f"Sentiment Score: {result['sentiment_score']:.2f}")
        suggestions = "\n".join(result['suggestions'])
        self.suggestions_label.config(text=f"Suggestions:\n{suggestions}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = RiskCalculatorApp(root, file_path=r"C:\Users\gvnsm\OneDrive\Desktop\manju\data_risk.csv")
    root.mainloop()

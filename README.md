# AI-Supply-Chain 
**AI-Driven Supply Chain Disruption Predictor and Inventory Optimization System
Description**

This project is dedicated to creating a smart, AI-driven solution that revolutionizes supply chain management. With real-time global monitoring and predictive analytics, it empowers businesses to proactively tackle disruptions while optimizing inventory levels. By leveraging cutting-edge LLMs like OpenAI GPT and Meta LLaMA, and many more

**Features**

1.**Data Collection**: Harness real-time insights through integration with APIs covering shipping, trade, logistics, and financial data.

2.**Predictive Modeling**: Utilize AI-powered algorithms to foresee supply chain disruptions and anticipate trends.

3.**Inventory Optimization**: Implement dynamic inventory strategies to cut costs and maximize operational efficiency.

4.**Real-Time Alerts**: Get instant notifications via Email, enabling swift and informed decisions.



**Task 1: Integrating Data APIs with Python**

What are APIs?

APIs (Application Programming Interfaces) enable seamless communication between software systems. In this project, APIs are used to collect real-time data on shipping, inventory, global trade, and financial trends.
Key Learnings:

**Authentication:** Securely use API keys.

**Rate Limiting:** Be aware of request limits.

**Response Formats:** Typically JSON or XML.

**Error Handling:** Handle timeouts and API failures.




**Task 2:Risk, and Supply Chain Analysis**

Fine-tuned a GPT-2 model on a custom dataset for text generation tasks. 

It includes dynamic padding, training checkpointing, and optimizer state saving. The trainingprocess uses PyTorch and the Hugging Face Transformers library.

**Risk Factors in the Medicine Supply Chain:**

**Raw Material Dependency:** 
Shortages or price fluctuations of key materials can halt production. 
Solution: Diversify suppliers and explore alternatives.

**Geopolitical Risks:** 
  Trade tensions or instability can disrupt supply chains. 
  Solution: Use multi-regional sourcing and adapt to trade policies.

**Natural Disasters:** 
  Disruptions from earthquakes, floods, or other disasters can halt production and deliveries. 
  Solution: Implement disaster recovery plans and diversify supply routes.

**Logistics and Transportation:** 
  Shipping delays and increased costs can cause stockouts. 
  Solution: Invest in logistics solutions, AI-powered forecasting, and local distribution hubs.

**Counterfeit and Fraud Risks:** 
  The entry of counterfeit drugs can harm patients and company reputation. 
  Solution: Use blockchain for traceability, serialization, and anti-counterfeit technologies.

**Regulatory and Compliance Risks:** 
  Variations in regulations can cause delays, fines, and recalls. 
  Solution: Establish compliance teams and collaborate with local authorities.

**Cybersecurity Threats:** 
  Cyberattacks on supply chain data can lead to data breaches and operational disruption. 
  Solution: Implement strong cybersecurity measures and real-time monitoring.

**Market Volatility:** 
  Sudden demand shifts can lead to inventory issues. 
  Solution: Use AI for demand forecasting, just-in-time inventory, and buffer stocks for high-demand items.


**Features**

**Dataset Loading**: Uses a JSONL dataset.

**Tokenizer Handling:** Implements dynamic padding.

**Training Pipeline:** Uses a DataLoader with custom batching.

**Checkpointing:** Saves model and optimizer states after each epoch.

**Resume Training:** Resumes from the last saved checkpoint.




**Task 3: Low Stock alert System**

**Data Loading and Preprocessing:**

Loads data from a CSV file.

Handles missing values by filling numeric columns with their mean and categorical columns with their mode.

Identifies categorical columns and returns the preprocessed data.

**Encoding Categorical Columns:**

Encodes categorical columns using LabelEncoder to convert them into numerical values.

**Model Training and Evaluation:**

Splits the data into features (X) and target (Risk_Factor).

Standardizes the data using StandardScaler and applies Principal Component Analysis (PCA) to reduce dimensionality.

Trains a RandomForestRegressor and evaluates the model using Root Mean Squared Error (RMSE).

**Low Stock Alerts Generation:**
Simulates stock levels based on the Economic_Factor column.

Identifies items with stock levels below a threshold (calculated as a percentage of the mean stock level).

Filters and returns items with low stock.

**Saving Results to CSV:**

Saves the filtered low stock items to a CSV file.





**Task 4: Low Stock Notifications to Email**

The system processes a dataset containing medicine details and risk factors, generates risk-based recommendations, and notifies relevant stakeholders via email.

**Key Features**

Data preprocessing for handling missing values and encoding categorical data.

Risk assessment based on supplier performance, economic factors, and transport status.

Automated suggestion generation for improving supply chain stability.

Email notifications with risk reports and CSV attachments.

**1. Data Processing Module**

**1.1 Load and Preprocess Data**

Function: load_and_preprocess_data(file_path)

Reads data from a CSV file.

Fills missing values in numerical columns with the mean.

Fills missing values in categorical columns with the mode.

Converts object-type columns to string format.

Returns the processed data along with categorical column names.

**1.2 Encode Categorical Columns**

Function: encode_categorical_columns(data, categorical_cols)

Encodes categorical columns using LabelEncoder.

Ensures that categorical values are transformed into numerical format for further analysis.

**2. Risk Analysis and Suggestions**

**Generate Risk-Based Suggestions**

Function: generate_suggestions(row)

Assesses the risk level of each medicine based on:

Discontinuation status

Supplier performance

Economic factors

Transport status

Overall risk factor

Assigns risk levels as High (🔴), Moderate (🟡), or Low (🟢).

Generates suggestions for inventory optimization and supply chain management.

Computes sentiment scores inversely proportional to risk levels.

**Email Notification System**

**Send Email Notifications**

Function: send_email(smtp_server, port, sender_email, sender_password, recipient_email, subject, body, attachment_file, file_path)

Configures SMTP settings for sending emails.

Attaches CSV reports containing medicine risk details.

Sends an alert to stakeholders regarding potential supply chain issues.

**Main Execution Flow**

Load and preprocess the dataset from /content/drive/MyDrive/data_risk.csv.

Encode categorical columns for analysis.

Accept user input for medicine name search.

Filter relevant medicine data and generate risk analysis reports.

Save analysis results to /content/drive/MyDrive/low_stock_alerts.csv.

Send an email notification containing risk details and the CSV report.

**Dependencies**

Ensure the following libraries are installed before executing the script:

pip install pandas numpy scikit-learn smtplib

**Security Considerations**

**Sensitive Credentials:**

Do not store plaintext passwords. Use environment variables or encrypted vaults.

**Data Privacy:**

Ensure that personal and proprietary medicine data is protected.

**Email Spam Prevention:**

Configure SMTP settings carefully to prevent emails from being marked as spam.

**Future Enhancements**

Implement an AI model for real-time prediction of supply chain disruptions.

Develop a web dashboard for interactive risk visualization.

Integrate with third-party ERP systems for seamless inventory adjustments.


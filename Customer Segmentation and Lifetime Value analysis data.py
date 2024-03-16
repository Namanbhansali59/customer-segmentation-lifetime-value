import pandas as pd
import numpy as np

# Define the number of customers
num_customers = 1000

# Generate random data for Recency, Frequency, Monetary, and Lifetime Value
np.random.seed(42)  # For reproducibility

recency = np.random.randint(1, 365, num_customers)  # Random number of days since last purchase
frequency = np.random.randint(1, 10, num_customers)  # Random number of purchases
monetary = np.random.randint(10, 1000, num_customers)  # Random purchase amount

# Calculate Lifetime Value based on Recency, Frequency, and Monetary
# Assuming Lifetime Value is directly proportional to the product of Frequency and Monetary
lifetime_value = frequency * monetary

# Create a DataFrame to store the generated data
data = pd.DataFrame({
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary,
    'Lifetime Value': lifetime_value
})

# Display the first few rows of the generated dataset
print(data.head())

# Save the dataset to a CSV file
data.to_csv('customer_transaction_data.csv', index=False)

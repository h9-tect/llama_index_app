# Llama Index Query Engine Streamlit App

## Overview

This project is a Streamlit application that allows users to interact with a custom query engine built using the `llama_index` library. Users can input their Llama Index API key and upload a PDF document, which is then processed and queried using various machine learning models and techniques.

## Features

- **Upload PDF Files**: Users can upload PDF files to be processed.
- **Llama Index API Integration**: The app requires users to input their Llama Index API key.
- **Custom Query Engine**: Utilizes models from the `llama_index` library to process and query the uploaded documents.
- **Streamlit Interface**: Provides a user-friendly interface for interacting with the query engine.

## Project Structure

llama_index_app/

├── app.py

├── settings.py

├── parsers.py

├── requirements.txt

└── README.md


## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- Streamlit
- GPU for quantization 

### Installation

1. **Clone the repository**:

```sh
   git clone https://github.com/h9-tect/llama_index_app.git
   cd llama_index_app
```

2. **Install dependencies:**:
 ```sh
   pip install -r requirements.txt
 ```
## Running the App

1. **Run the app**:
```sh
   streamlit run app.py
```

## Usage

- Enter Llama Index API Key: Input your Llama Index API key in the provided text box.
- Upload PDF Document: Use the file uploader to select and upload a PDF document.
- Query the Document: Enter your queries in the text input box to interact with the processed document.



## Files Description

- app.py: Main application file that sets up the Streamlit interface and handles user inputs.
- settings.py: Contains configuration settings and functions to initialize the models and load documents.
- parsers.py: Defines custom parsers to process the uploaded PDF documents.
- requirements.txt: Lists all the dependencies required for the project.

## Contributing

Feel free to fork this repository and submit pull requests. For significant changes, please open an issue first to discuss what you would like to change.
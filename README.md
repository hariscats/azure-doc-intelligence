# azure-doc-intelligence

Three scripts that extract structured data from PDFs using Azure Document Intelligence.

## Scripts

**extract_layout.py** — pulls out paragraphs, tables, and checkboxes using Azure's prebuilt layout model.

**extract_document.py** — same as above but also detects key-value pairs, barcodes, handwriting, and languages.

**custom_extract_model.py** — train and use a custom model on your own labeled documents. Supports train, analyze, info, and list modes.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Azure credentials.

## Usage

```
python extract_layout.py path/to/file.pdf
python extract_document.py path/to/file.pdf
python custom_extract_model.py path/to/file.pdf

python custom_extract_model.py --train
python custom_extract_model.py --list
python custom_extract_model.py --info
```

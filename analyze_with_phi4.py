"""
Analyze documents with Phi-4-multimodal-instruct via Azure AI Foundry.

Complements extract_layout.py by taking the structured output from
Document Intelligence and sending it to Phi-4 for deeper analysis:
  - Summarization
  - Key-value extraction
  - Question answering

Usage:
  python analyze_with_phi4.py <path_to_pdf>
  python analyze_with_phi4.py <path_to_pdf> --question "What is the total amount due?"
"""

import os
import sys
import json
import argparse
import requests
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient


# ────────────────────────────────────────────────────────────
# 1.  DOCUMENT INTELLIGENCE — Extract structured text
# ────────────────────────────────────────────────────────────

def extract_layout_text(file_path: str, endpoint: str, key: str) -> dict:
    """Run the prebuilt-layout model and return structured sections."""

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", body=f)
    result = poller.result()

    # --- paragraphs (with roles) ---
    paragraphs_text = []
    if result.paragraphs:
        for p in result.paragraphs:
            role = p.role or "text"
            content = p.content.replace("\n", " ")
            if role in ("pageHeader", "pageFooter", "pageNumber"):
                continue
            if role == "title":
                paragraphs_text.append(f"# {content}")
            elif role == "sectionHeading":
                paragraphs_text.append(f"## {content}")
            else:
                paragraphs_text.append(content)

    # --- tables (markdown format) ---
    tables_text = []
    if result.tables:
        for idx, table in enumerate(result.tables, 1):
            grid = [["" for _ in range(table.column_count)]
                    for _ in range(table.row_count)]
            for cell in table.cells:
                grid[cell.row_index][cell.column_index] = (
                    cell.content.replace("\n", " ").strip()
                )
            md_rows = []
            for row_idx, row in enumerate(grid):
                md_rows.append("| " + " | ".join(row) + " |")
                if row_idx == 0:
                    md_rows.append("| " + " | ".join("---" for _ in row) + " |")
            tables_text.append(f"### Table {idx}\n" + "\n".join(md_rows))

    # --- selection marks ---
    marks_text = []
    for page in result.pages:
        if page.selection_marks:
            for m in page.selection_marks:
                state = "☑" if m.state == "selected" else "☐"
                marks_text.append(f"{state} (page {page.page_number})")

    return {
        "paragraphs": "\n".join(paragraphs_text),
        "tables": "\n\n".join(tables_text),
        "selection_marks": ", ".join(marks_text) if marks_text else "None",
        "page_count": len(result.pages),
        "model_id": result.model_id,
    }


# ────────────────────────────────────────────────────────────
# 2.  PHI-4 MULTIMODAL — Analyze extracted content
# ────────────────────────────────────────────────────────────

def call_phi4(endpoint: str, key: str, system_prompt: str, user_prompt: str) -> str:
    """Send a chat completion request to Phi-4-multimodal-instruct."""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    payload = {
        "model": "Phi-4-multimodal-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)

    if not response.ok:
        print(f"  ERROR {response.status_code}: {response.text}")
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


# ────────────────────────────────────────────────────────────
# 3.  ANALYSIS MODES
# ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert document analyst. You receive structured text "
    "extracted from a document via Azure Document Intelligence. "
    "Provide clear, accurate, and well-organized analysis."
)


def build_summarize_prompt(extracted: dict) -> str:
    return (
        "Below is the structured content extracted from a document.\n\n"
        f"**Document text:**\n{extracted['paragraphs']}\n\n"
        f"**Tables:**\n{extracted['tables']}\n\n"
        f"**Selection marks:** {extracted['selection_marks']}\n\n"
        "Please provide:\n"
        "1. A concise summary of the document (2-3 paragraphs).\n"
        "2. Key data points or figures mentioned.\n"
        "3. Any action items or important dates.\n"
    )


def build_question_prompt(extracted: dict, question: str) -> str:
    return (
        "Below is the structured content extracted from a document.\n\n"
        f"**Document text:**\n{extracted['paragraphs']}\n\n"
        f"**Tables:**\n{extracted['tables']}\n\n"
        f"**Selection marks:** {extracted['selection_marks']}\n\n"
        f"**Question:** {question}\n\n"
        "Answer the question based only on the document content above. "
        "If the answer is not in the document, say so."
    )


# ────────────────────────────────────────────────────────────
# 4.  MAIN
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract a document with Document Intelligence, "
                    "then analyze it with Phi-4-multimodal-instruct.",
    )
    parser.add_argument("file", help="Path to the PDF or image to analyze.")
    parser.add_argument(
        "--question", "-q",
        help="Ask a specific question about the document. "
             "If omitted, the script produces a summary.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    # Load credentials
    load_dotenv()
    di_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    di_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    phi4_endpoint = os.getenv("PHI4_ENDPOINT")
    phi4_key = os.getenv("PHI4_KEY")

    if not di_endpoint or not di_key:
        print("ERROR: Missing Document Intelligence credentials in .env")
        sys.exit(1)
    if not phi4_endpoint or not phi4_key:
        print("ERROR: Missing Phi-4 credentials in .env")
        sys.exit(1)

    # ── Step 1: Extract with Document Intelligence ──
    print("=" * 60)
    print("  STEP 1 — Extracting document with Document Intelligence")
    print("=" * 60)
    print(f"  File: {args.file}\n")

    extracted = extract_layout_text(args.file, di_endpoint, di_key)

    print(f"  ✓ Extracted {extracted['page_count']} page(s)")
    print(f"    Model: {extracted['model_id']}")
    print(f"    Paragraphs: {len(extracted['paragraphs'].splitlines())} lines")
    tables_count = extracted["tables"].count("### Table") if extracted["tables"] else 0
    print(f"    Tables: {tables_count}")
    print(f"    Selection marks: {extracted['selection_marks']}")

    # ── Step 2: Analyze with Phi-4 ──
    mode = "Question" if args.question else "Summary"
    print()
    print("=" * 60)
    print(f"  STEP 2 — {mode} with Phi-4-multimodal-instruct")
    print("=" * 60)

    if args.question:
        prompt = build_question_prompt(extracted, args.question)
    else:
        prompt = build_summarize_prompt(extracted)

    print("  Sending to Phi-4 …")
    answer = call_phi4(phi4_endpoint, phi4_key, SYSTEM_PROMPT, prompt)

    print()
    print("-" * 60)
    print(f"  PHI-4 {mode.upper()}")
    print("-" * 60)
    print()
    print(answer)
    print()


if __name__ == "__main__":
    main()

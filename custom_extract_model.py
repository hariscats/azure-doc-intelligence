import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import (
    DocumentIntelligenceClient,
    DocumentIntelligenceAdministrationClient,
)
from azure.ai.documentintelligence.models import (
    BuildDocumentModelRequest,
    AzureBlobContentSource,
    DocumentBuildMode,
)


def print_banner(mode):
    """Print the mode-specific banner."""
    banners = {
        "train": (
            "CUSTOM MODEL — Train on Labeled ASQR Data",
            "Builds a neural model from labeled PDFs in Azure Blob Storage",
        ),
        "analyze": (
            "CUSTOM MODEL — Extract ASQR-Specific Fields",
            "Uses a trained custom model to extract structured fields",
        ),
        "info": (
            "CUSTOM MODEL — Model Details",
            "Displays field schema, doc types, and training info",
        ),
        "list": (
            "CUSTOM MODEL — List All Models",
            "Shows all custom (non-prebuilt) models in the resource",
        ),
    }
    title, subtitle = banners[mode]
    print("=" * 60)
    print(f"  {title}")
    print(f"  {subtitle}")
    print("=" * 60)


def load_credentials():
    """Load and validate Azure credentials from .env."""
    load_dotenv()
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not key:
        print("ERROR: Missing credentials in .env file.")
        exit(1)

    return endpoint, key


def get_admin_client(endpoint, key):
    """Create an administration client for model management."""
    return DocumentIntelligenceAdministrationClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )


def get_analyze_client(endpoint, key):
    """Create a client for document analysis."""
    return DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )


# ================================================================
# --train mode
# ================================================================
def do_train(endpoint, key):
    """Train a custom neural model from labeled data in blob storage."""
    print_banner("train")

    sas_url = os.getenv("AZURE_BLOB_CONTAINER_SAS_URL")
    if not sas_url:
        print("\nERROR: AZURE_BLOB_CONTAINER_SAS_URL not set in .env file.")
        exit(1)

    model_id = os.getenv("CUSTOM_MODEL_ID") or f"asqr-custom-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Prerequisites reminder
    print("\n--- PREREQUISITES ---\n")
    print("  Before training, ensure you have:")
    print("  1. Uploaded ASQR PDFs to the blob container")
    print("  2. Labeled them in Document Intelligence Studio")
    print("     https://documentintelligence.ai.azure.com/studio")
    print("  3. SAS URL has read + list permissions")
    print(f"\n  Model ID:    {model_id}")
    print(f"  Build mode:  neural")
    print(f"  Container:   {sas_url.split('?')[0]}")

    confirm = input("\n  Proceed with training? (y/N): ").strip().lower()
    if confirm != "y":
        print("\n  Training cancelled.")
        return

    print("\n--- TRAINING ---\n")
    print("  Starting model build (neural models can take 10-30+ minutes)...\n")

    admin_client = get_admin_client(endpoint, key)

    poller = admin_client.begin_build_document_model(
        BuildDocumentModelRequest(
            model_id=model_id,
            build_mode=DocumentBuildMode.NEURAL,
            azure_blob_source=AzureBlobContentSource(container_url=sas_url),
        )
    )

    model = poller.result()

    # ============================================================
    # TRAINING RESULTS
    # ============================================================
    print("--- TRAINING RESULTS ---\n")
    print(f"  Model ID:         {model.model_id}")
    print(f"  Status:           {model.status if hasattr(model, 'status') else 'succeeded'}")
    print(f"  Created:          {model.created_date_time}")
    print(f"  API version:      {model.api_version}")

    if model.doc_types:
        for doc_type_name, doc_type in model.doc_types.items():
            print(f"\n  Document type: {doc_type_name}")
            if doc_type.field_schema:
                print(f"  Fields ({len(doc_type.field_schema)}):")
                for field_name, field_info in doc_type.field_schema.items():
                    field_type = field_info.get("type", "unknown") if isinstance(field_info, dict) else getattr(field_info, "type", "unknown")
                    print(f"    - {field_name} ({field_type})")

    print(f"\n--- NEXT STEP ---\n")
    print(f"  Add this to your .env file:")
    print(f"  CUSTOM_MODEL_ID={model.model_id}\n")
    print(f"  Then run:  python custom_extract_model.py <pdf_path>\n")


# ================================================================
# --analyze mode (default)
# ================================================================
def do_analyze(endpoint, key, file_path):
    """Analyze a document using the trained custom model."""
    print_banner("analyze")

    model_id = os.getenv("CUSTOM_MODEL_ID")
    if not model_id:
        print("\nERROR: CUSTOM_MODEL_ID not set in .env file.")
        print("  Train a model first with: python custom_extract_model.py --train")
        exit(1)

    if not os.path.exists(file_path):
        print(f"\nERROR: File not found: {file_path}")
        exit(1)

    print(f"\n  Model:     {model_id}")
    print(f"  Analyzing: {file_path}\n")

    client = get_analyze_client(endpoint, key)

    with open(file_path, "rb") as document_file:
        poller = client.begin_analyze_document(model_id, body=document_file)
    result = poller.result()

    # Field categories for organized output
    categories = {
        "Document Metadata": [
            "DocumentNumber", "Revision", "EffectiveDate", "Function",
            "Title", "Date", "Rev",
        ],
        "Member Applicability": [
            "MemberName", "Abbreviation", "ChapterApplicability",
            "Member", "Applicability",
        ],
        "Supplier Forms": [
            "FormNumber", "FormName", "SupplierForm",
        ],
    }

    # ============================================================
    # EXTRACTED DOCUMENTS
    # ============================================================
    if result.documents:
        for doc_idx, document in enumerate(result.documents):
            print(f"--- DOCUMENT {doc_idx + 1} (type: {document.doc_type}, "
                  f"confidence: {document.confidence:.0%}) ---\n")

            if not document.fields:
                print("  No fields extracted.\n")
                continue

            # Group fields by category
            categorized = {cat: {} for cat in categories}
            uncategorized = {}

            for field_name, field in document.fields.items():
                placed = False
                for cat, keywords in categories.items():
                    if any(kw.lower() in field_name.lower() for kw in keywords):
                        categorized[cat][field_name] = field
                        placed = True
                        break
                if not placed:
                    uncategorized[field_name] = field

            # Print each category
            for cat, fields in categorized.items():
                if not fields:
                    continue
                print(f"  [{cat}]")
                for field_name, field in fields.items():
                    print_field(field_name, field, indent=4)
                print()

            if uncategorized:
                print("  [Other Fields]")
                for field_name, field in uncategorized.items():
                    print_field(field_name, field, indent=4)
                print()
    else:
        print("  No documents extracted. The model may not match this document.\n")

    # ============================================================
    # SUMMARY
    # ============================================================
    total_fields = 0
    if result.documents:
        for doc in result.documents:
            if doc.fields:
                total_fields += len(doc.fields)

    print("--- SUMMARY ---\n")
    print(f"  Documents:   {len(result.documents) if result.documents else 0}")
    print(f"  Fields:      {total_fields}")
    print(f"  Model:       {model_id} (API {result.api_version})")


def print_field(name, field, indent=4):
    """Print a single extracted field with confidence."""
    prefix = " " * indent
    confidence = field.confidence if field.confidence else 0

    if field.type == "array" and field.value_array:
        print(f"{prefix}{name}: (list, {len(field.value_array)} items, {confidence:.0%})")
        for i, item in enumerate(field.value_array):
            if item.type == "object" and item.value_object:
                print(f"{prefix}  [{i + 1}]")
                for sub_name, sub_field in item.value_object.items():
                    print_field(sub_name, sub_field, indent=indent + 6)
            else:
                val = item.content if item.content else "(empty)"
                print(f"{prefix}  [{i + 1}] {val}")
    elif field.type == "object" and field.value_object:
        print(f"{prefix}{name}: (object, {confidence:.0%})")
        for sub_name, sub_field in field.value_object.items():
            print_field(sub_name, sub_field, indent=indent + 2)
    else:
        val = field.content if field.content else field.value_string if field.value_string else "(empty)"
        print(f"{prefix}{name}: {val}  ({confidence:.0%})")


# ================================================================
# --info mode
# ================================================================
def do_info(endpoint, key):
    """Display details about the trained custom model."""
    print_banner("info")

    model_id = os.getenv("CUSTOM_MODEL_ID")
    if not model_id:
        print("\nERROR: CUSTOM_MODEL_ID not set in .env file.")
        exit(1)

    print(f"\n  Fetching model: {model_id}\n")

    admin_client = get_admin_client(endpoint, key)
    model = admin_client.get_model(model_id)

    print("--- MODEL DETAILS ---\n")
    print(f"  Model ID:    {model.model_id}")
    print(f"  Status:      {model.status if hasattr(model, 'status') else 'N/A'}")
    print(f"  Created:     {model.created_date_time}")
    print(f"  API version: {model.api_version}")
    print(f"  Description: {model.description or '(none)'}")

    if model.doc_types:
        print(f"\n--- DOCUMENT TYPES ({len(model.doc_types)}) ---\n")
        for doc_type_name, doc_type in model.doc_types.items():
            print(f"  Type: {doc_type_name}")
            if doc_type.field_schema:
                print(f"  Fields ({len(doc_type.field_schema)}):")
                for field_name, field_info in doc_type.field_schema.items():
                    field_type = field_info.get("type", "unknown") if isinstance(field_info, dict) else getattr(field_info, "type", "unknown")
                    desc = ""
                    if isinstance(field_info, dict) and field_info.get("description"):
                        desc = f" — {field_info['description']}"
                    print(f"    - {field_name} ({field_type}){desc}")
            if doc_type.field_confidence:
                print(f"  Field confidence:")
                for field_name, conf in doc_type.field_confidence.items():
                    print(f"    - {field_name}: {conf:.0%}")
            print()
    else:
        print("\n  No document types defined.\n")


# ================================================================
# --list mode
# ================================================================
def do_list(endpoint, key):
    """List all custom (non-prebuilt) models in the resource."""
    print_banner("list")
    print()

    admin_client = get_admin_client(endpoint, key)
    models = admin_client.list_models()

    print("--- CUSTOM MODELS ---\n")
    count = 0
    for model in models:
        # Skip prebuilt models
        if model.model_id.startswith("prebuilt-"):
            continue
        count += 1
        status = model.status if hasattr(model, "status") else "N/A"
        print(f"  {count}. {model.model_id}")
        print(f"     Status:  {status}")
        print(f"     Created: {model.created_date_time}")
        if model.description:
            print(f"     Desc:    {model.description}")
        print()

    if count == 0:
        print("  No custom models found.\n")

    print(f"--- SUMMARY ---\n")
    print(f"  Custom models: {count}\n")


# ================================================================
# CLI entry point
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train and use custom Document Intelligence models for ASQR extraction."
    )
    parser.add_argument("file", nargs="?",
                        help="Path to PDF file (required for analyze mode)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true",
                       help="Train a new custom model from labeled blob data")
    group.add_argument("--info", action="store_true",
                       help="Show details about the trained model")
    group.add_argument("--list", action="store_true",
                       help="List all custom models in the resource")

    args = parser.parse_args()

    endpoint, key = load_credentials()

    if args.train:
        do_train(endpoint, key)
    elif args.info:
        do_info(endpoint, key)
    elif args.list:
        do_list(endpoint, key)
    else:
        if not args.file:
            parser.error("A PDF file path is required for analyze mode.\n"
                         "Usage: python custom_extract_model.py <path_to_pdf>")
        do_analyze(endpoint, key, args.file)


if __name__ == "__main__":
    main()

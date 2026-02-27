import os
import sys
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    ClassifyDocumentRequest,
    AzureBlobContentSource,
)


def main():
    # 1. Load Environment Variables
    load_dotenv()
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    classifier_id = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_CLASSIFIER_ID")

    if not endpoint or not key:
        print("ERROR: Missing credentials in .env file.")
        exit(1)

    if not classifier_id:
        print("ERROR: Missing AZURE_DOCUMENT_INTELLIGENCE_CLASSIFIER_ID in .env file.")
        print("\nTo use classification, you must first build a custom classifier")
        print("in Document Intelligence Studio or via the API.")
        print("See: https://learn.microsoft.com/azure/ai-services/document-intelligence/concept-custom-classifier")
        exit(1)

    # 2. Get file path from CLI argument
    if len(sys.argv) < 2:
        print("Usage: python classify_document.py <path_to_pdf>")
        exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        exit(1)

    # 3. Authenticate the Client
    client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    print("=" * 60)
    print("  CUSTOM CLASSIFIER — Document Classification")
    print("  Demonstrates: classifying documents into categories,")
    print("  confidence scores, multi-document splitting")
    print("=" * 60)
    print(f"\nClassifier ID: {classifier_id}")
    print(f"Analyzing:     {file_path}\n")

    # 4. Classify Document
    with open(file_path, "rb") as document_file:
        poller = client.begin_classify_document(
            classifier_id,
            body=document_file,
        )
    result = poller.result()

    # ============================================================
    # CLASSIFICATION RESULTS
    # ============================================================
    print("--- CLASSIFICATION RESULTS ---\n")

    if result.documents:
        for idx, document in enumerate(result.documents, start=1):
            doc_type = document.doc_type if document.doc_type else "Unknown"
            confidence = document.confidence if document.confidence else 0.0

            # Confidence bar for visual clarity
            bar_length = int(confidence * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)

            print(f"  Document #{idx}")
            print(f"    Type:       {doc_type}")
            print(f"    Confidence: {confidence:.2%}  [{bar}]")

            # Show page ranges for this classified segment
            if document.bounding_regions:
                pages = sorted(set(br.page_number for br in document.bounding_regions))
                if len(pages) == 1:
                    print(f"    Pages:      {pages[0]}")
                else:
                    print(f"    Pages:      {pages[0]}–{pages[-1]} ({len(pages)} pages)")

            # Show spans (character offsets) if available
            if document.spans:
                for span in document.spans:
                    print(f"    Span:       offset={span.offset}, length={span.length}")

            print()
    else:
        print("  No documents were classified.\n")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("--- SUMMARY ---\n")
    print(f"  Classifier:       {classifier_id}")
    print(f"  Documents found:  {len(result.documents) if result.documents else 0}")

    if result.documents:
        # Group by document type
        type_counts = {}
        for doc in result.documents:
            doc_type = doc.doc_type or "Unknown"
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        print(f"  Document types:")
        for doc_type, count in sorted(type_counts.items()):
            print(f"    - {doc_type}: {count}")

        # Average confidence
        confidences = [d.confidence for d in result.documents if d.confidence]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"  Avg confidence:   {avg_confidence:.2%}")

    print(f"  Model:            {result.model_id} (API {result.api_version})")


if __name__ == "__main__":
    main()
import os
import sys
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature


def main():
    # 1. Load Environment Variables
    load_dotenv()
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not key:
        print("ERROR: Missing credentials in .env file.")
        exit(1)

    # 2. Get file path from CLI argument
    if len(sys.argv) < 2:
        print("Usage: python extract_document.py <path_to_pdf>")
        exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        exit(1)

    # 3. Authenticate the Client
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    print("=" * 60)
    print("  ADD-ON FEATURES — Enhanced Data Extraction")
    print("  Demonstrates: key-value pairs, font styles,")
    print("  barcode detection, language detection")
    print("=" * 60)
    print(f"\nAnalyzing: {file_path}\n")

    # 4. Analyze Document using prebuilt-layout with add-on features
    #    The prebuilt-document model was retired in newer API versions.
    #    Instead, we use prebuilt-layout with optional features enabled to
    #    extract key-value pairs, font styles, barcodes, and languages.
    features = [
        DocumentAnalysisFeature.KEY_VALUE_PAIRS,
        DocumentAnalysisFeature.STYLE_FONT,
        DocumentAnalysisFeature.BARCODES,
        DocumentAnalysisFeature.LANGUAGES,
    ]
    with open(file_path, "rb") as document_file:
        poller = client.begin_analyze_document(
            "prebuilt-layout", body=document_file, features=features
        )
    result = poller.result()

    # ============================================================
    # KEY-VALUE PAIRS (Form Fields)
    # ============================================================
    print("--- KEY-VALUE PAIRS (Form Fields) ---\n")
    if result.key_value_pairs:
        for kv in result.key_value_pairs:
            key_text = kv.key.content.strip() if kv.key else "(no key)"
            value_text = kv.value.content.strip() if kv.value else "(empty)"
            confidence = kv.confidence if kv.confidence else 0
            print(f"  {key_text}: {value_text}  ({confidence:.0%})")
    else:
        print("  No key-value pairs found.")

    # ============================================================
    # CONTENT STYLES (Handwritten vs Printed)
    # ============================================================
    print("\n--- CONTENT STYLES ---\n")
    if result.styles:
        handwritten = [s for s in result.styles if s.is_handwritten]
        printed = [s for s in result.styles if s.is_handwritten is not None and not s.is_handwritten]
        if handwritten:
            print(f"  Handwritten: {len(handwritten)} region(s) detected")
        if printed:
            print(f"  Printed:     {len(printed)} region(s) detected")
        if not handwritten and not printed:
            print("  No handwriting/print classification detected.")
    else:
        print("  No style information detected.")

    # ============================================================
    # BARCODES
    # ============================================================
    print("\n--- BARCODES ---\n")
    total_barcodes = 0
    for page in result.pages:
        if page.barcodes:
            total_barcodes += len(page.barcodes)
            for barcode in page.barcodes:
                print(f"  Page {page.page_number}: [{barcode.kind}] {barcode.value} "
                      f"({barcode.confidence:.0%})")
    if total_barcodes == 0:
        print("  No barcodes found.")

    # ============================================================
    # LANGUAGES (aggregated by locale)
    # ============================================================
    print("\n--- DETECTED LANGUAGES ---\n")
    if result.languages:
        lang_summary = {}
        for lang in result.languages:
            locale = lang.locale
            if locale not in lang_summary:
                lang_summary[locale] = {"count": 0, "max_confidence": 0}
            lang_summary[locale]["count"] += len(lang.spans)
            lang_summary[locale]["max_confidence"] = max(
                lang_summary[locale]["max_confidence"], lang.confidence
            )
        for locale, info in sorted(lang_summary.items(), key=lambda x: -x[1]["count"]):
            print(f"  {locale}: {info['count']} span(s), "
                  f"best confidence: {info['max_confidence']:.0%}")
    else:
        print("  No language information detected.")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n--- SUMMARY ---\n")
    print(f"  Pages:            {len(result.pages)}")
    print(f"  Key-value pairs:  {len(result.key_value_pairs) if result.key_value_pairs else 0}")
    print(f"  Barcodes:         {total_barcodes}")
    print(f"  Languages:        {len(lang_summary) if result.languages else 0}")
    print(f"  Model:            {result.model_id} (API {result.api_version})")
    print(f"  Add-on features:  {', '.join(f.value for f in features)}")


if __name__ == "__main__":
    main()

"""
OCR & Handwriting Recognition — prebuilt-read model

Uses the prebuilt-read model (optimized for text-heavy and handwritten
documents) to extract every word from a scanned image or PDF, then
separates handwritten content from printed text using style detection.

Demonstrates:
  - High-fidelity OCR on scanned / photographed documents
  - Handwriting vs. printed classification per text span
  - Per-word and per-line confidence scores
  - Language detection across the document

Usage:
  python ocr_handwriting.py <path_to_image_or_pdf>
  python ocr_handwriting.py scan.jpg
  python ocr_handwriting.py form.pdf
"""

import os
import sys
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature


def confidence_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def classify_spans(styles) -> dict:
    """
    Build a set of character offsets that are handwritten vs. printed
    from the styles array returned by DI.
    """
    handwritten_offsets = set()
    printed_offsets = set()

    if not styles:
        return {"handwritten": handwritten_offsets, "printed": printed_offsets}

    for style in styles:
        if style.is_handwritten is None:
            continue
        target = handwritten_offsets if style.is_handwritten else printed_offsets
        if style.spans:
            for span in style.spans:
                for offset in range(span.offset, span.offset + span.length):
                    target.add(offset)

    return {"handwritten": handwritten_offsets, "printed": printed_offsets}


def word_is_handwritten(word, span_map: dict) -> bool | None:
    """
    Check whether a word falls inside a handwritten span.
    Returns True (handwritten), False (printed), or None (unknown).
    """
    if not word.span:
        return None
    start = word.span.offset
    end = start + word.span.length
    hw_count = sum(1 for o in range(start, end) if o in span_map["handwritten"])
    pr_count = sum(1 for o in range(start, end) if o in span_map["printed"])
    if hw_count > pr_count:
        return True
    elif pr_count > hw_count:
        return False
    return None


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
        print("Usage: python ocr_handwriting.py <path_to_image_or_pdf>")
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
    print("  OCR & HANDWRITING RECOGNITION")
    print("  Model: prebuilt-read (optimized for text + handwriting)")
    print("  Add-ons: STYLE_FONT, LANGUAGES")
    print("=" * 60)
    print(f"\nAnalyzing: {file_path}\n")

    # 4. Analyze with prebuilt-read + style and language add-ons
    #    prebuilt-read is purpose-built for OCR-heavy scenarios:
    #    scanned paper, faxes, photographed documents, handwritten notes.
    features = [
        DocumentAnalysisFeature.STYLE_FONT,
        DocumentAnalysisFeature.LANGUAGES,
    ]

    with open(file_path, "rb") as document_file:
        poller = client.begin_analyze_document(
            "prebuilt-read", body=document_file, features=features
        )
    result = poller.result()

    # ── Build handwritten/printed span map ──
    span_map = classify_spans(result.styles)

    # ============================================================
    # HANDWRITING vs. PRINTED DETECTION
    # ============================================================
    print("--- HANDWRITING vs. PRINTED DETECTION ---\n")

    if result.styles:
        hw_styles = [s for s in result.styles if s.is_handwritten]
        pr_styles = [s for s in result.styles
                     if s.is_handwritten is not None and not s.is_handwritten]

        hw_chars = len(span_map["handwritten"])
        pr_chars = len(span_map["printed"])
        total_chars = hw_chars + pr_chars or 1

        print(f"  Handwritten regions: {len(hw_styles)}")
        print(f"  Printed regions:     {len(pr_styles)}")
        print(f"  Handwritten chars:   {hw_chars} ({hw_chars/total_chars:.0%})")
        print(f"  Printed chars:       {pr_chars} ({pr_chars/total_chars:.0%})")

        if hw_chars > 0:
            print(f"\n  [ {'HANDWRITING DETECTED':^40} ]")
        else:
            print(f"\n  [ {'NO HANDWRITING — all printed':^40} ]")
    else:
        print("  No style information available (style detection not supported")
        print("  for this file type or region).")

    # ============================================================
    # PER-PAGE OCR RESULTS
    # ============================================================
    print(f"\n--- PER-PAGE OCR RESULTS ({len(result.pages)} pages) ---\n")

    total_words = 0
    total_hw_words = 0
    all_confidences = []

    for page in result.pages:
        words = page.words or []
        lines = page.lines or []
        total_words += len(words)

        # Classify words on this page
        hw_words = []
        pr_words = []
        unknown_words = []

        for word in words:
            all_confidences.append(word.confidence or 0)
            hw = word_is_handwritten(word, span_map)
            if hw is True:
                hw_words.append(word)
            elif hw is False:
                pr_words.append(word)
            else:
                unknown_words.append(word)

        total_hw_words += len(hw_words)

        page_conf = (sum(w.confidence or 0 for w in words) / len(words)) if words else 0

        print(f"  Page {page.page_number}")
        print(f"    Dimensions:  {page.width} x {page.height} {page.unit or ''}")
        print(f"    Lines:       {len(lines)}")
        print(f"    Words:       {len(words)}")
        print(f"    Avg conf:    {page_conf:.1%}  {confidence_bar(page_conf)}")
        print(f"    Handwritten: {len(hw_words)} word(s)")
        print(f"    Printed:     {len(pr_words)} word(s)")
        if unknown_words:
            print(f"    Unclassified:{len(unknown_words)} word(s)")

        # ── Show lines with handwriting tags ──
        if lines:
            print(f"\n    --- Lines (page {page.page_number}) ---\n")
            for line_idx, line in enumerate(lines):
                text = line.content

                # Check if this line is handwritten
                line_words = [w for w in words
                              if w.span and line.spans
                              and any(w.span.offset >= s.offset
                                      and w.span.offset < s.offset + s.length
                                      for s in line.spans)]

                hw_count = sum(1 for w in line_words if word_is_handwritten(w, span_map) is True)
                total_in_line = len(line_words) or 1

                if hw_count / total_in_line > 0.5:
                    tag = "✍ HW"
                else:
                    tag = "  PR"

                # Line-level confidence (average of its words)
                line_conf = (sum(w.confidence or 0 for w in line_words) /
                             len(line_words)) if line_words else 0

                # Truncate long lines for display
                display_text = text[:90]
                if len(text) > 90:
                    display_text += " …"

                print(f"    {tag} {line_conf:.0%} │ {display_text}")

        print()

    # ============================================================
    # FONT STYLE DETAILS
    # ============================================================
    print("--- FONT STYLE DETAILS ---\n")

    if result.styles:
        font_styles = [s for s in result.styles
                       if getattr(s, "font_style", None) or getattr(s, "font_weight", None)]
        if font_styles:
            seen = set()
            for style in font_styles:
                font = getattr(style, "similar_font_family", None) or "unknown"
                weight = getattr(style, "font_weight", None) or "normal"
                fstyle = getattr(style, "font_style", None) or "normal"
                key = (font, weight, fstyle)
                if key not in seen:
                    seen.add(key)
                    conf = style.confidence or 0
                    print(f"  Font: {font}, Weight: {weight}, "
                          f"Style: {fstyle}, Confidence: {conf:.0%}")
        else:
            print("  No detailed font styles detected.")
    else:
        print("  No style data available.")

    # ============================================================
    # DETECTED LANGUAGES
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
    # HANDWRITTEN TEXT (consolidated)
    # ============================================================
    if total_hw_words > 0:
        print("\n--- HANDWRITTEN TEXT (extracted) ---\n")
        print("  The following text was identified as handwritten:\n")

        for page in result.pages:
            words = page.words or []
            hw_on_page = [w for w in words if word_is_handwritten(w, span_map) is True]
            if hw_on_page:
                # Reconstruct handwritten text from consecutive words
                hw_text = " ".join(w.content for w in hw_on_page)
                avg_conf = sum(w.confidence or 0 for w in hw_on_page) / len(hw_on_page)
                print(f"  Page {page.page_number} ({len(hw_on_page)} words, "
                      f"avg confidence: {avg_conf:.0%}):")
                # Wrap long text
                words_list = hw_text.split()
                line = "    "
                for word in words_list:
                    if len(line) + len(word) + 1 > 80:
                        print(line)
                        line = "    "
                    line += word + " "
                if line.strip():
                    print(line)
                print()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("--- SUMMARY ---\n")
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    print(f"  Pages:             {len(result.pages)}")
    print(f"  Total words:       {total_words}")
    print(f"  Handwritten words: {total_hw_words}")
    print(f"  Printed words:     {total_words - total_hw_words}")
    print(f"  Avg word conf:     {avg_conf:.1%}  {confidence_bar(avg_conf)}")
    print(f"  Languages:         {len(result.languages) if result.languages else 0}")
    print(f"  Model:             {result.model_id} (API {result.api_version})")
    print(f"  Add-on features:   {', '.join(f.value for f in features)}")


if __name__ == "__main__":
    main()

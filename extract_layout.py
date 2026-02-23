import os
import sys
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient


def print_formatted_table(table):
    """Helper function to print tables in a clean, readable grid."""
    print(f"\n[Table: {table.row_count} rows x {table.column_count} columns]")

    # Create an empty grid
    grid = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]

    # Populate the grid with cell content
    for cell in table.cells:
        clean_content = cell.content.replace('\n', ' ').strip()
        grid[cell.row_index][cell.column_index] = clean_content

    # Print the grid
    for row_idx, row in enumerate(grid):
        formatted_row = " | ".join(f"{col[:27] + '...' if len(col) > 30 else col:<30}" for col in row)
        print(f"| {formatted_row} |")

        if row_idx == 0:
            separator = "-+-".join("-" * 30 for _ in range(table.column_count))
            print(f"|-{separator}-|")
    print()


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
        print("Usage: python extract_layout.py <path_to_pdf>")
        exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        exit(1)

    # 3. Authenticate the Client
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    print("=" * 60)
    print("  LAYOUT MODEL — Visual Structure & Document Organization")
    print("  Demonstrates: paragraphs, semantic roles, tables,")
    print("  selection marks (checkboxes), page structure")
    print("=" * 60)
    print(f"\nAnalyzing: {file_path}\n")

    # 4. Analyze Document
    with open(file_path, "rb") as document_file:
        poller = client.begin_analyze_document("prebuilt-layout", body=document_file)
    result = poller.result()

    total_selection_marks = 0

    # ============================================================
    # DOCUMENT STRUCTURE (Paragraphs with semantic roles)
    # ============================================================
    print("--- DOCUMENT STRUCTURE (Semantic Roles) ---\n")

    if result.paragraphs:
        for paragraph in result.paragraphs:
            role = paragraph.role if paragraph.role else "text"
            content = paragraph.content.replace('\n', ' ')

            if role == "title":
                print(f"\n# {content.upper()}")
                print("=" * (len(content) + 2))
            elif role == "sectionHeading":
                print(f"\n## {content}")
            elif role in ["pageHeader", "pageFooter", "pageNumber"]:
                continue
            else:
                print(f"{content}")

    # ============================================================
    # DATA TABLES
    # ============================================================
    print("\n\n--- TABLES ---")
    if result.tables:
        for table in result.tables:
            print_formatted_table(table)
    else:
        print("No tables found.\n")

    # ============================================================
    # SELECTION MARKS (Checkboxes)
    # ============================================================
    print("--- SELECTION MARKS (Checkboxes) ---\n")
    for page in result.pages:
        if page.selection_marks:
            marks = page.selection_marks
            total_selection_marks += len(marks)
            selected = sum(1 for m in marks if m.state == "selected")
            print(f"  Page {page.page_number}: {len(marks)} checkbox(es) "
                  f"— {selected} checked, {len(marks) - selected} unchecked")
    if total_selection_marks == 0:
        print("  No selection marks found.")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n--- SUMMARY ---\n")
    print(f"  Pages:            {len(result.pages)}")
    print(f"  Paragraphs:       {len(result.paragraphs) if result.paragraphs else 0}")
    print(f"  Tables:           {len(result.tables) if result.tables else 0}")
    print(f"  Selection marks:  {total_selection_marks}")
    print(f"  Model:            {result.model_id} (API {result.api_version})")


if __name__ == "__main__":
    main()
